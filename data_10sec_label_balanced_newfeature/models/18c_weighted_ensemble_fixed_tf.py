# 18c_weighted_ensemble_fixed_tf.py
# Ensemble (CNN + Transformer + InceptionTime) with FIXED weights and per-fold thr-opt (accuracy)
# Config chosen to match the ~63% accuracy runs you observed.

import os, json, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D, Dense,
                                     Dropout, Layer, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============== Paths ==============
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "18c_weighted_ensemble_fixed_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

# ============== Data ==============
X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ============== Config (matches the ~63% runs) ==============
WIN, STR        = 500, 200
N_SPLITS        = 6
VAL_FILE_RATIO  = 0.35
SEEDS           = [7, 17, 23]
TOPK_K          = 5
NEG_BOOST       = 1.4
LR_INIT         = 1e-3
EPOCHS          = 80
BATCH_SIZE      = 16
L2_REG          = 1e-5
DROPOUT_BLOCK   = 0.30
DROPOUT_HEAD    = 0.35

# Threshold search range
THR_MIN, THR_MAX, THR_STEPS = 0.40, 0.80, 41

# Aggregation mode for window->file: median is what worked best
AGG_MODE   = "median"   # ("median" or "mean" or "trimmed")
AGG_TRIM_Q = 0.10       # only used if AGG_MODE == "trimmed"

# Fixed ensemble weights (cnn, transformer, inception)
W_CNN, W_TRF, W_INC = 0.20, 0.50, 0.30

# ============== Utilities ==============
np.random.seed(42); tf.random.set_seed(42); random.seed(42)

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

Xw, Yw, Gw, Fw = make_windows(X, Y, G, WIN, STR)
print(f"[WINDOWS] Xw={Xw.shape}, Yw={Yw.shape}, Gw={Gw.shape}, Fw={Fw.shape}", flush=True)

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

def agg_file(probs, file_ids):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    out = {}
    for f, vals in d.items():
        v = np.array(vals, dtype=float)
        if AGG_MODE == "median":
            out[f] = float(np.median(v))
        elif AGG_MODE == "trimmed":
            q = AGG_TRIM_Q
            lo, hi = np.quantile(v, q), np.quantile(v, 1-q)
            v2 = v[(v>=lo) & (v<=hi)]
            out[f] = float(np.mean(v2)) if len(v2)>0 else float(np.mean(v))
        else:
            out[f] = float(np.mean(v))
    return out

def choose_files_for_validation_stratified(train_file_ids, y_file_dict, ratio, seed=0):
    rng = np.random.default_rng(seed)
    files0 = [f for f in train_file_ids if y_file_dict[f]==0]
    files1 = [f for f in train_file_ids if y_file_dict[f]==1]
    n_val = max(1, int(math.ceil(len(train_file_ids)*ratio)))
    # split proportional to class counts
    n1 = max(1, int(round(n_val * (len(files1)/(len(files0)+len(files1)+1e-9)))))
    n0 = max(1, n_val - n1)
    val0 = rng.choice(files0, size=min(n0,len(files0)), replace=False).tolist() if files0 else []
    val1 = rng.choice(files1, size=min(n1,len(files1)), replace=False).tolist() if files1 else []
    val = set(val0 + val1)
    remaining = [f for f in train_file_ids if f not in val]
    while len(val) < n_val and remaining:
        val.add(remaining.pop())
    return val

def choose_thr_for_accuracy(file_probs, file_true, tmin, tmax, steps):
    grid = np.linspace(tmin, tmax, steps)
    files = list(file_probs.keys())
    y = np.array([file_true[f] for f in files], dtype=int)
    p = np.array([file_probs[f] for f in files], dtype=float)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        pred = (p > t).astype(int)
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc

class TopKSaver(tf.keras.callbacks.Callback):
    def __init__(self, model, k=TOPK_K):
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

# ============== Models ==============
def build_cnn(input_shape):
    m = Sequential([
        Input(shape=input_shape),
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
        Dense(64, activation='relu'), Dropout(DROPOUT_HEAD),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

class PositionalEncoding(Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def call(self, x):
        # x: (B, T, d)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        d = tf.shape(x)[2]
        pos = tf.cast(tf.range(T)[:, None], tf.float32)
        i   = tf.cast(tf.range(self.d_model)[None, :], tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2*(i//2))/tf.cast(self.d_model, tf.float32))
        angle_rads  = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        coses = tf.cos(angle_rads[:, 1::2])
        pe = tf.reshape(tf.concat([sines, coses], axis=-1), (T, self.d_model))
        pe = pe[None, :, :]                                       # (1,T,d)
        pe = tf.cast(pe, x.dtype)
        return x + pe[:, :T, :d]
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg

def transformer_block(x, d_model=64, num_heads=4, ffn_dim=128, dropout=0.2):
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(x, x)
    x = Add()([x, attn_out]); x = BatchNormalization()(x)
    f = Dense(ffn_dim, activation='relu')(x); f = Dropout(dropout)(f)
    f = Dense(d_model)(f)
    x = Add()([x, f]); x = BatchNormalization()(x)
    return x

def build_transformer(input_shape, d_model=64, heads=4, layers=1, ffn=128, dropout=0.2):
    inp = Input(shape=input_shape)
    x = Conv1D(d_model, 1, padding='same')(inp)
    x = PositionalEncoding(d_model)(x)
    for _ in range(layers):
        x = transformer_block(x, d_model=d_model, num_heads=heads, ffn_dim=ffn, dropout=dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x); x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

# InceptionTime (light)
def inception_module(x, nb_filters=32, kernel_sizes=(9,19,39), bottleneck=32, dropout=0.2):
    if bottleneck and int(x.shape[-1]) > 1:
        x_in = Conv1D(bottleneck, 1, padding='same', activation='linear')(x)
    else:
        x_in = x
    convs = []
    for k in kernel_sizes:
        convs.append(Conv1D(nb_filters, k, padding='same', activation='relu')(x_in))
    pool = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    pool = Conv1D(nb_filters, 1, padding='same', activation='relu')(pool)
    x_out = tf.keras.layers.concatenate(convs + [pool], axis=-1)
    x_out = BatchNormalization()(x_out)
    x_out = Activation('relu')(x_out)
    x_out = Dropout(dropout)(x_out)
    return x_out

def build_inceptiontime(input_shape, n_modules=3, nb_filters=32, bottleneck=32, dropout=0.2):
    inp = Input(shape=input_shape)
    x = inp
    for _ in range(n_modules):
        x = inception_module(x, nb_filters=nb_filters, bottleneck=bottleneck, dropout=dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x); x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def fit_seed_ensemble(build_fn, Xtr, Ytr, Xva, Xte, class_weight):
    va_preds, te_preds = [], []
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)
        model = build_fn((Xtr.shape[1], Xtr.shape[2]))

        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_K)

        model.fit(Xtr, Ytr,
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(Xva, Yva),
                  class_weight=class_weight,
                  verbose=0,
                  callbacks=[es, rlr, topk])
        topk.set_topk_weights()

        va_preds.append(model.predict(Xva, verbose=0).ravel())
        te_preds.append(model.predict(Xte, verbose=0).ravel())

    return np.mean(va_preds, axis=0), np.mean(te_preds, axis=0)

def platt_calibrate(val_file_probs, val_file_true, test_file_probs):
    files = sorted(val_file_probs.keys())
    vX = np.array([val_file_probs[f] for f in files])[:, None]
    vy = np.array([val_file_true[f] for f in files], dtype=int)
    if len(np.unique(vy)) < 2:
        # nothing to calibrate
        return test_file_probs
    lr = LogisticRegression(max_iter=1000)
    lr.fit(vX, vy)
    test_files = sorted(test_file_probs.keys())
    tX = np.array([test_file_probs[f] for f in test_files])[:, None]
    tCal = lr.predict_proba(tX)[:,1]
    return {f: float(p) for f, p in zip(test_files, tCal)}

# ============== CV loop ==============
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
fold = 1

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)
    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    # choose val files (stratified by file label)
    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}
    val_files = choose_files_for_validation_stratified(train_files, y_file, VAL_FILE_RATIO, seed=42+fold)

    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # standardize per fold
    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    # class weights
    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # ==== Train each base model (seed-ensemble+TopK) ====
    # CNN
    Yva = Yva_in  # needed in fit_seed_ensemble callback call
    va_cnn_w, te_cnn_w = fit_seed_ensemble(build_cnn, Xtr_in, Ytr_in, Xva_in, Xte, cw)
    f_va_cnn = agg_file(va_cnn_w, Fva_in); f_te_cnn = agg_file(te_cnn_w, Fte)

    # Transformer
    va_trf_w, te_trf_w = fit_seed_ensemble(
        lambda shape: build_transformer(shape, d_model=64, heads=4, layers=1, ffn=128, dropout=0.2),
        Xtr_in, Ytr_in, Xva_in, Xte, cw)
    f_va_trf = agg_file(va_trf_w, Fva_in); f_te_trf = agg_file(te_trf_w, Fte)

    # InceptionTime
    va_inc_w, te_inc_w = fit_seed_ensemble(
        lambda shape: build_inceptiontime(shape, n_modules=3, nb_filters=32, bottleneck=32, dropout=0.2),
        Xtr_in, Ytr_in, Xva_in, Xte, cw)
    f_va_inc = agg_file(va_inc_w, Fva_in); f_te_inc = agg_file(te_inc_w, Fte)

    # ==== Optional: Platt calibration per model (helps stability) ====
    val_true = {int(fi): int(Y[fi]) for fi in set(Fva_in.tolist())}
    f_te_cnn_cal = platt_calibrate(f_va_cnn, val_true, f_te_cnn)
    f_te_trf_cal = platt_calibrate(f_va_trf, val_true, f_te_trf)
    f_te_inc_cal = platt_calibrate(f_va_inc, val_true, f_te_inc)

    # Also calibrate validation (to pick thr in calibrated space)
    f_va_cnn_cal = platt_calibrate(f_va_cnn, val_true, f_va_cnn)
    f_va_trf_cal = platt_calibrate(f_va_trf, val_true, f_va_trf)
    f_va_inc_cal = platt_calibrate(f_va_inc, val_true, f_va_inc)

    # ==== Weighted ensemble of calibrated scores ====
    # Build dict of ensemble probs for val and test
    all_va_files = sorted(list(set(list(f_va_cnn_cal.keys()) + list(f_va_trf_cal.keys()) + list(f_va_inc_cal.keys()))))
    ens_va = {f: W_CNN*f_va_cnn_cal.get(f,0.5) + W_TRF*f_va_trf_cal.get(f,0.5) + W_INC*f_va_inc_cal.get(f,0.5) for f in all_va_files}

    all_te_files = sorted(list(set(list(f_te_cnn_cal.keys()) + list(f_te_trf_cal.keys()) + list(f_te_inc_cal.keys()))))
    ens_te = {f: W_CNN*f_te_cnn_cal.get(f,0.5) + W_TRF*f_te_trf_cal.get(f,0.5) + W_INC*f_te_inc_cal.get(f,0.5) for f in all_te_files}

    # threshold selection by accuracy on validation
    thr, acc_val = choose_thr_for_accuracy(ens_va, {f:int(Y[f]) for f in ens_va.keys()}, THR_MIN, THR_MAX, THR_STEPS)
    print(f"[FOLD {fold}] Picked thr={thr:.3f} | Acc(val)={acc_val:.3f}")

    # evaluate on test files
    t_files = sorted(ens_te.keys())
    t_y = np.array([int(Y[f]) for f in t_files], dtype=int)
    t_p = np.array([ens_te[f] for f in t_files], dtype=float)
    y_pred = (t_p > thr).astype(int)

    acc  = accuracy_score(t_y, y_pred)
    prec = precision_score(t_y, y_pred, zero_division=0)
    rec  = recall_score(t_y, y_pred, zero_division=0)
    f1   = f1_score(t_y, y_pred, zero_division=0)

    rows.append({"fold":fold, "thr":float(thr),
                 "acc":acc, "precision":prec, "recall":rec, "f1":f1})

    print(f"[FOLD {fold}] Test: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    fold += 1

# ============== Summary & Save ==============
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "per_fold_metrics.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = {
    "config":{
        "WIN":WIN, "STR":STR, "Seeds":SEEDS, "TopK":TOPK_K,
        "NEG_BOOST":NEG_BOOST, "AGG_MODE":AGG_MODE, "AGG_TRIM_Q":AGG_TRIM_Q,
        "VAL_FILE_RATIO":VAL_FILE_RATIO, "THR_RANGE":[THR_MIN, THR_MAX, THR_STEPS],
        "weights_fixed":{"cnn":W_CNN, "transformer":W_TRF, "inception":W_INC},
        "models":{
            "cnn":"3xConv-BN-ReLU + GAP",
            "transformer":"1x encoder (d_model=64, heads=4, ffn=128, drop=0.2)",
            "inceptiontime":"3 modules, nf=32, bottleneck=32, drop=0.2"
        }
    },
    "Mean±STD":{
        "Accuracy":[float(mean["acc"]), float(std["acc"])],
        "Precision":[float(mean["precision"]), float(std["precision"])],
        "Recall":[float(mean["recall"]), float(std["recall"])],
        "F1":[float(mean["f1"]), float(std["f1"])],
    },
    "thr_list":[float(t) for t in df["thr"].tolist()]
}

with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Pretty print table
print("\n[PER-FOLD RESULTS]")
print(f"{'fold':>4} {'acc':>6} {'precision':>10} {'recall':>7} {'f1':>6} {'thr':>6}")
for r in rows:
    print(f"{r['fold']:>4} {r['acc']:.3f} {r['precision']:.3f} {r['recall']:.3f} {r['f1']:.3f} {r['thr']:.3f}")

print("\n[MEAN ± STD]")
print(f"Accuracy : {mean['acc']:.4f} ± {std['acc']:.4f}")
print(f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}")
print(f"Recall   : {mean['recall']:.4f} ± {std['recall']:.4f}")
print(f"F1       : {mean['f1']:.4f} ± {std['f1']:.4f}")

print("Saved to:", RESULT_DIR, flush=True)
