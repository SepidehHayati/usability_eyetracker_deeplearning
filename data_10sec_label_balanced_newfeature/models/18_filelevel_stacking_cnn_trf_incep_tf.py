# 18_filelevel_stacking_cnn_trf_incep_tf.py
# Ensemble at FILE-LEVEL:
# - Train three window models (CNN, Transformer, InceptionTime) per fold
# - Aggregate window probs -> file probs (median or trimmed-mean)
# - STACK: [p_cnn, p_trf, p_incep] -> LogisticRegression (fit on val-files, eval on test-files)
# - Also report simple average ensemble as baseline
#
# Settings are aligned with your best runs so far.

import os, math, random, json, numpy as np, pandas as pd
from collections import defaultdict

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation, MaxPooling1D,
                                     GlobalAveragePooling1D, Dense, Dropout, Add, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "18_filelevel_stacking_cnn_trf_incep_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))     # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ======================= Config (unified) =======================
# Windowing
WIN, STR = 500, 200

# CV & training
N_SPLITS     = 6
SEEDS        = [7, 17, 23]
EPOCHS       = 80
BATCH_SIZE   = 16
LR_INIT      = 1e-3
L2_REG       = 1e-5
DROPOUT_BLOCK= 0.30
DROPOUT_HEAD = 0.35
RANDOM_STATE = 42
VAL_FILE_RATIO = 0.35

# Class weight tweak to reduce FP (boost class 0)
NEG_BOOST    = 1.4

# Aggregation at window->file
AGG_MODE     = "median"      # {"median", "trimmed"}
AGG_TRIM_Q   = 0.10          # only used if AGG_MODE == "trimmed"

# Threshold grids
THR_MIN, THR_MAX, THR_STEPS = 0.40, 0.80, 41   # for val threshold search

# Transformer (lite, fast)
TRF_D_MODEL  = 64
TRF_HEADS    = 4
TRF_LAYERS   = 1
TRF_FFN      = 128
TRF_DROPOUT  = 0.20

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

# ======================= Utils =======================
def make_windows(X, Y, G, win=WIN, stride=STR):
    Xw, Yw, Gw, Fw = [], [], [], []
    for i in range(X.shape[0]):
        xi = X[i]
        for s in range(0, T - win + 1, stride):
            Xw.append(xi[s:s+win, :])
            Yw.append(Y[i]); Gw.append(G[i]); Fw.append(i)
    return (np.asarray(Xw, dtype=np.float32),
            np.asarray(Yw, dtype=np.int64),
            np.asarray(Gw, dtype=np.int64),
            np.asarray(Fw, dtype=np.int64))

def standardize_per_fold(train_arr, *others):
    t_len, c = train_arr.shape[1], train_arr.shape[2]
    sc = StandardScaler()
    sc.fit(train_arr.reshape(-1, c))
    out = []
    for arr in (train_arr,) + others:
        A2 = arr.reshape(-1, c)
        A2 = sc.transform(A2)
        out.append(A2.reshape(arr.shape[0], t_len, c))
    return out

def compute_class_weights(y, neg_boost=1.0):
    classes = np.unique(y)
    cnts = np.array([(y==c).sum() for c in classes], dtype=np.float32)
    n, k = len(y), len(classes)
    w = {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, cnts)}
    if 0 in w:
        w[0] = w[0] * neg_boost
    return w

def agg_probs_file_level(probs, file_ids, mode=AGG_MODE, trim_q=AGG_TRIM_Q):
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    out = {}
    for f, lst in d.items():
        arr = np.array(lst, dtype=float)
        if mode == "median":
            out[f] = float(np.median(arr))
        elif mode == "trimmed":
            lo = np.quantile(arr, trim_q)
            hi = np.quantile(arr, 1.0-trim_q)
            arr2 = arr[(arr>=lo)&(arr<=hi)]
            out[f] = float(arr2.mean()) if len(arr2)>0 else float(arr.mean())
        else:
            out[f] = float(arr.mean())
    return out

def choose_files_for_validation_stratified(train_file_ids, y_file_dict, ratio, seed=0):
    rng = np.random.default_rng(seed)
    files0 = [f for f in train_file_ids if y_file_dict[f]==0]
    files1 = [f for f in train_file_ids if y_file_dict[f]==1]
    n_val = max(1, int(math.ceil(len(train_file_ids)*ratio)))
    # proportional split
    n1 = max(1, int(round(n_val * (len(files1)/(len(files0)+len(files1)+1e-9)))))
    n0 = max(1, n_val - n1)
    val0 = rng.choice(files0, size=min(n0,len(files0)), replace=False).tolist() if files0 else []
    val1 = rng.choice(files1, size=min(n1,len(files1)), replace=False).tolist() if files1 else []
    val = set(val0 + val1)
    remaining = [f for f in train_file_ids if f not in val]
    while len(val) < n_val and remaining:
        val.add(remaining.pop())
    return val

def choose_thr_for_accuracy(file_probs: dict, file_true: dict,
                            tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS):
    grid = np.linspace(tmin, tmax, steps)
    files = list(file_probs.keys())
    y_true = np.array([file_true[f] for f in files], dtype=int)
    p_vec  = np.array([file_probs[f] for f in files], dtype=float)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        y_hat = (p_vec > t).astype(int)
        acc = accuracy_score(y_true, y_hat)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc

class TopKSaver(tf.keras.callbacks.Callback):
    def __init__(self, model, k=5):
        super().__init__()
        self.model_ref = model
        self.k = k
        self.snaps = []
    def on_epoch_end(self, epoch, logs=None):
        vloss = float(logs.get('val_loss', np.inf))
        self.snaps.append((vloss, [w.copy() for w in self.model_ref.get_weights()]))
    def set_topk_weights(self):
        if not self.snaps: return
        snaps_sorted = sorted(self.snaps, key=lambda x: x[0])
        top = snaps_sorted[:min(self.k, len(self.snaps))]
        avg = [np.mean([w[i] for (_vl, w) in top], axis=0) for i in range(len(top[0][1]))]
        self.model_ref.set_weights(avg)

# ======================= Models =======================
def build_cnn(input_shape):
    m = Sequential([
        tf.keras.Input(shape=input_shape),
        Conv1D(48, 7, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROPOUT_BLOCK),

        Conv1D(64, 5, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROPOUT_BLOCK),

        Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROPOUT_BLOCK),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_HEAD),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def call(self, x):
        # x: (B, T, D)
        B = tf.shape(x)[0]
        Tt = tf.shape(x)[1]
        i = tf.range(Tt, dtype=tf.float32)[:, None]     # (T,1)
        j = tf.range(self.d_model, dtype=tf.float32)[None, :]  # (1,D)
        angle_rates = 1.0 / tf.pow(10000.0, (2.0*(tf.floor(j/2.0)))/tf.cast(self.d_model, tf.float32))
        angles = i * angle_rates  # (T,D)
        sines = tf.sin(angles[:, 0::2])
        coses = tf.cos(angles[:, 1::2])
        pe = tf.reshape(tf.stack([sines, coses], axis=-1), (Tt, -1))
        pe = pe[:, :self.d_model]
        pe = tf.expand_dims(pe, axis=0)  # (1,T,D)
        return x + pe

def transformer_encoder(x, d_model, num_heads, ffn, dropout):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x)
    x1 = tf.keras.layers.Dropout(dropout)(attn)
    x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + x1)

    f = tf.keras.layers.Dense(ffn, activation='relu')(x2)
    f = tf.keras.layers.Dropout(dropout)(f)
    f = tf.keras.layers.Dense(d_model)(f)
    x3 = tf.keras.layers.Dropout(dropout)(f)
    out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2 + x3)
    return out

def build_transformer(input_shape):
    inp = Input(shape=input_shape)
    x = Conv1D(TRF_D_MODEL, 1, padding='same')(inp)
    x = PositionalEncoding(TRF_D_MODEL)(x)
    for _ in range(TRF_LAYERS):
        x = transformer_encoder(x, TRF_D_MODEL, TRF_HEADS, TRF_FFN, TRF_DROPOUT)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT_HEAD)(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def inception_module(x, nb_filters):
    # kernel sizes
    b1 = Conv1D(nb_filters, 1, padding='same', activation='relu')(x)

    b2 = Conv1D(nb_filters, 1, padding='same', activation='relu')(x)
    b2 = Conv1D(nb_filters, 3, padding='same', activation='relu')(b2)

    b3 = Conv1D(nb_filters, 1, padding='same', activation='relu')(x)
    b3 = Conv1D(nb_filters, 5, padding='same', activation='relu')(b3)

    b4 = MaxPooling1D(3, strides=1, padding='same')(x)
    b4 = Conv1D(nb_filters, 1, padding='same', activation='relu')(b4)

    x = Concatenate()([b1, b2, b3, b4])
    x = BatchNormalization()(x)
    return x

def build_inceptiontime(input_shape):
    inp = Input(shape=input_shape)
    x = inception_module(inp, 16)
    x = inception_module(x, 16)
    x = inception_module(x, 16)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT_HEAD)(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ======================= Train & get probs (one model) =======================
def train_and_get_file_probs(model_builder, Xtr_in, Ytr_in, Ftr_in, Xva_in, Yva_in, Fva_in, Xte, Fte):
    # class weights
    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)

    val_win_probs_list, te_win_probs_list = [], []
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = model_builder((Xtr_in.shape[1], Xtr_in.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=5)

        model.fit(Xtr_in, Ytr_in,
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(Xva_in, Yva_in),
                  class_weight=cw, verbose=0,
                  callbacks=[es, rlr, topk])

        topk.set_topk_weights()

        val_win_probs_list.append(model.predict(Xva_in, verbose=0).ravel())
        te_win_probs_list.append(model.predict(Xte,   verbose=0).ravel())

    val_win_probs = np.mean(val_win_probs_list, axis=0)
    te_win_probs  = np.mean(te_win_probs_list,  axis=0)

    val_file_probs = agg_probs_file_level(val_win_probs, Fva_in, mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    te_file_probs  = agg_probs_file_level(te_win_probs,  Fte,    mode=AGG_MODE, trim_q=AGG_TRIM_Q)

    return val_file_probs, te_file_probs

# ======================= Main CV loop =======================
Xw, Yw, Gw, Fw = make_windows(X, Y, G, WIN, STR)
print(f"[WINDOWS] Xw={Xw.shape}, Yw={Yw.shape}, Gw={Gw.shape}, Fw={Fw.shape}", flush=True)

gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
fold = 1

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)
    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    # pick val-files (stratified by file label)
    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}
    val_files = choose_files_for_validation_stratified(train_files, y_file, VAL_FILE_RATIO,
                                                       seed=RANDOM_STATE + fold)
    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # standardize per-fold
    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    print("[CLASS WEIGHTS]", compute_class_weights(Ytr_in, neg_boost=NEG_BOOST), flush=True)

    # --- CNN ---
    v_cnn, t_cnn = train_and_get_file_probs(build_cnn,          Xtr_in, Ytr_in, Ftr_in, Xva_in, Yva_in, Fva_in, Xte, Fte)
    # --- Transformer ---
    v_trf, t_trf = train_and_get_file_probs(build_transformer,  Xtr_in, Ytr_in, Ftr_in, Xva_in, Yva_in, Fva_in, Xte, Fte)
    # --- InceptionTime ---
    v_inc, t_inc = train_and_get_file_probs(build_inceptiontime,Xtr_in, Ytr_in, Ftr_in, Xva_in, Yva_in, Fva_in, Xte, Fte)

    # Build stacking matrices (VAL -> fit, TEST -> eval)
    val_files_sorted = sorted(list(val_files))
    Xv = np.column_stack([
        np.array([v_cnn[f] for f in val_files_sorted], dtype=float),
        np.array([v_trf[f] for f in val_files_sorted], dtype=float),
        np.array([v_inc[f] for f in val_files_sorted], dtype=float),
    ])
    yv = np.array([Y[f] for f in val_files_sorted], dtype=int)

    test_files_sorted = sorted(list(set(Fte)))  # unique files in test fold
    Xt = np.column_stack([
        np.array([t_cnn[f] for f in test_files_sorted], dtype=float),
        np.array([t_trf[f] for f in test_files_sorted], dtype=float),
        np.array([t_inc[f] for f in test_files_sorted], dtype=float),
    ])
    yt = np.array([Y[f] for f in test_files_sorted], dtype=int)

    # ===== Baseline: simple average ensemble =====
    val_avg = Xv.mean(axis=1)
    test_avg = Xt.mean(axis=1)
    thr_avg, _ = choose_thr_for_accuracy({i: val_avg[k] for k,i in enumerate(val_files_sorted)},
                                         {i: Y[i] for i in val_files_sorted})
    y_pred_avg = (test_avg > thr_avg).astype(int)
    acc_avg  = accuracy_score(yt, y_pred_avg)
    prec_avg = precision_score(yt, y_pred_avg, zero_division=0)
    rec_avg  = recall_score(yt, y_pred_avg, zero_division=0)
    f1_avg   = f1_score(yt, y_pred_avg, zero_division=0)

    rows.append({"fold":fold,"mode":"avg","thr":float(thr_avg),
                 "acc":acc_avg,"precision":prec_avg,"recall":rec_avg,"f1":f1_avg})

    # ===== Stacking: Logistic Regression =====
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xv, yv)
    val_lr_prob = lr.predict_proba(Xv)[:,1]
    test_lr_prob= lr.predict_proba(Xt)[:,1]

    thr_lr, _ = choose_thr_for_accuracy({i:val_lr_prob[k] for k,i in enumerate(val_files_sorted)},
                                        {i: Y[i] for i in val_files_sorted})
    y_pred_lr = (test_lr_prob > thr_lr).astype(int)
    acc_lr  = accuracy_score(yt, y_pred_lr)
    prec_lr = precision_score(yt, y_pred_lr, zero_division=0)
    rec_lr  = recall_score(yt, y_pred_lr, zero_division=0)
    f1_lr   = f1_score(yt, y_pred_lr, zero_division=0)

    rows.append({"fold":fold,"mode":"stack_lr","thr":float(thr_lr),
                 "acc":acc_lr,"precision":prec_lr,"recall":rec_lr,"f1":f1_lr})

    print(f"[FOLD {fold}] AVG thr={thr_avg:.3f} | Acc={acc_avg:.3f} Prec={prec_avg:.3f} Rec={rec_avg:.3f} F1={f1_avg:.3f}")
    print(f"[FOLD {fold}] LR  thr={thr_lr:.3f} | Acc={acc_lr:.3f} Prec={prec_lr:.3f} Rec={rec_lr:.3f} F1={f1_lr:.3f}")

    fold += 1

# ======================= Save & Summary =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "stacking_metrics_per_fold.csv"), index=False)

def summarize(df, mode):
    sub = df[df["mode"]==mode]
    mean = sub.mean(numeric_only=True); std = sub.std(numeric_only=True)
    return {
        "mode": mode,
        "acc_mean": float(mean["acc"]), "acc_std": float(std["acc"]),
        "prec_mean": float(mean["precision"]), "prec_std": float(std["precision"]),
        "rec_mean": float(mean["recall"]), "rec_std": float(std["recall"]),
        "f1_mean": float(mean["f1"]), "f1_std": float(std["f1"]),
        "thr_list": [float(t) for t in sub["thr"].tolist()],
    }

summary = {
    "config": {
        "WIN": WIN, "STR": STR, "Seeds": SEEDS, "TopK": 5,
        "NEG_BOOST": NEG_BOOST, "AGG_MODE": AGG_MODE, "AGG_TRIM_Q": AGG_TRIM_Q,
        "VAL_FILE_RATIO": VAL_FILE_RATIO,
        "THR_RANGE": [THR_MIN, THR_MAX, THR_STEPS],
        "Transformer": {"d_model": TRF_D_MODEL, "heads": TRF_HEADS, "layers": TRF_LAYERS,
                        "ffn": TRF_FFN, "dropout": TRF_DROPOUT},
    },
    "avg": summarize(df, "avg"),
    "stack_lr": summarize(df, "stack_lr"),
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[SUMMARY]")
print(json.dumps(summary, indent=2))
print("Saved to:", RESULT_DIR, flush=True)
