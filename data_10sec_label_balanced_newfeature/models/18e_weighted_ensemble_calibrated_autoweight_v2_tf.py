# 18e_weighted_ensemble_calibrated_autoweight_v2_tf.py
# هدف: انسمبل وزن‌دار (CNN + Transformer + InceptionTime) با کالیبراسیون Platt
#      + انتخاب خودکار وزن‌ها و آستانه بر اساس دقت فایل‌سطح روی ولیدیشن.
# نکته: این نسخه باگ validation_data را رفع کرده و خروجی نهایی را به صورت جدول هم چاپ می‌کند.

import os, math, random, json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, Add, Lambda, LayerNormalization)
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "18e_weighted_ensemble_calibrated_autoweight_v2_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ======================= Config =======================
WIN, STR        = 500, 200
N_SPLITS        = 6
EPOCHS          = 80
BATCH_SIZE      = 16
LR_INIT         = 1e-3
L2_REG          = 1e-5
DROPOUT_BLOCK   = 0.30
DROPOUT_HEAD    = 0.35
RANDOM_STATE    = 42

VAL_FILE_RATIO  = 0.35
THR_MIN, THR_MAX, THR_STEPS = 0.40, 0.80, 101

SEEDS           = [7, 17, 23]
TOPK_K          = 5
NEG_BOOST       = 1.4

AGG_MODE        = "median"  # یا "trimmed"
AGG_TRIM_Q      = 0.10

USE_PLATT_CAL   = True

WEIGHT_CANDIDATES = [
    (0.20, 0.50, 0.30),
    (0.25, 0.50, 0.25),
    (0.30, 0.40, 0.30),
    (0.30, 0.50, 0.20),
    (0.40, 0.40, 0.20),
    (0.40, 0.30, 0.30),
    (0.20, 0.40, 0.40),
    (0.33, 0.34, 0.33),
]

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

# ======================= Windowing =======================
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

# ======================= Utils =======================
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
    if 0 in w:  # punish FP
        w[0] = w[0] * neg_boost
    return w

def agg_probs_file_level(probs, file_ids, mode=AGG_MODE, trim_q=AGG_TRIM_Q):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    out = {}
    for f, ps in d.items():
        arr = np.array(ps, dtype=float)
        if mode == "median":
            out[f] = float(np.median(arr))
        elif mode == "trimmed":
            lo = np.quantile(arr, trim_q)
            hi = np.quantile(arr, 1.0 - trim_q)
            trimmed = arr[(arr>=lo) & (arr<=hi)]
            out[f] = float(np.mean(trimmed)) if trimmed.size>0 else float(np.mean(arr))
        else:  # mean
            out[f] = float(np.mean(arr))
    return out

def choose_files_for_validation_stratified(train_file_ids, y_file_dict, ratio, seed=0):
    rng = np.random.default_rng(seed)
    files0 = [f for f in train_file_ids if y_file_dict[f]==0]
    files1 = [f for f in train_file_ids if y_file_dict[f]==1]
    n_val = max(1, int(math.ceil(len(train_file_ids)*ratio)))
    n1 = max(1, int(round(n_val * (len(files1)/(len(files0)+len(files1)+1e-9)))))
    n0 = max(1, n_val - n1)
    val0 = rng.choice(files0, size=min(n0,len(files0)), replace=False).tolist() if files0 else []
    val1 = rng.choice(files1, size=min(n1,len(files1)), replace=False).tolist() if files1 else []
    val = set(val0 + val1)
    remaining = [f for f in train_file_ids if f not in val]
    while len(val) < n_val and remaining:
        val.add(remaining.pop())
    return val

def thr_grid_best_acc(p_true_dict, p_pred_dict, tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS):
    grid = np.linspace(tmin, tmax, steps)
    files = sorted(p_pred_dict.keys())
    y_true = np.array([p_true_dict[f] for f in files], dtype=int)
    p_vec  = np.array([p_pred_dict[f] for f in files], dtype=float)
    best = (0.5, -1.0, 0.0)  # thr, acc, prec (tie-breaker)
    for t in grid:
        y_hat = (p_vec > t).astype(int)
        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        if acc > best[1] or (acc == best[1] and prec > best[2]):
            best = (float(t), float(acc), float(prec))
    return best  # (thr, acc, prec)

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

# ======================= Models =======================
def build_cnn(input_shape):
    model = Sequential([
        tf.keras.Input(shape=input_shape),

        Conv1D(48, kernel_size=7, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(pool_size=2), Dropout(DROPOUT_BLOCK),

        Conv1D(64, kernel_size=5, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(pool_size=2), Dropout(DROPOUT_BLOCK),

        Conv1D(64, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(pool_size=2), Dropout(DROPOUT_BLOCK),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_HEAD),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def transformer_block(x, d_model=64, num_heads=4, ffn_dim=128, dropout=0.2):
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(x, x)
    x = Add()([x, attn_out]); x = LayerNormalization()(x)
    ffn = Sequential([Dense(ffn_dim, activation='relu'),
                      Dropout(dropout),
                      Dense(d_model)])(x)
    x = Add()([x, ffn]); x = LayerNormalization()(x)
    return x

def build_transformer(input_shape, d_model=64, heads=4, layers=1, ffn=128, dropout=0.2):
    inp = Input(shape=input_shape)
    x = Conv1D(d_model, kernel_size=1, padding='same')(inp)  # projection
    for _ in range(layers):
        x = transformer_block(x, d_model, heads, ffn, dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(DROPOUT_HEAD)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def inception_module(x, nb_filters=32, bottleneck=32, kernel_sizes=(9,19,39), use_bottleneck=True):
    if use_bottleneck:
        x_in = Conv1D(bottleneck, kernel_size=1, padding='same', use_bias=False)(x)
    else:
        x_in = x
    conv_list = []
    for k in kernel_sizes:
        conv_list.append(Conv1D(nb_filters, kernel_size=k, padding='same', use_bias=False)(x_in))
    pool = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    pool = Conv1D(nb_filters, kernel_size=1, padding='same', use_bias=False)(pool)
    x = tf.keras.layers.Concatenate(axis=-1)(conv_list + [pool])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_inceptiontime(input_shape, n_modules=3, nb_filters=32, bottleneck=32, drop=0.2):
    inp = Input(shape=input_shape)
    x = inp
    for i in range(n_modules):
        x = inception_module(x, nb_filters=nb_filters, bottleneck=bottleneck)
        x = inception_module(x, nb_filters=nb_filters, bottleneck=bottleneck)
        x_res = Conv1D(x.shape[-1], kernel_size=1, padding='same')(inp if i==0 else x_res)
        x = Add()([x, x_res])
        x = Activation('relu')(x)
        x = Dropout(drop)(x)
        x_res = x
    x = GlobalAveragePooling1D()(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======================= Fit helper (FIX APPLIED) =======================
def fit_predict_model(build_fn, Xtr, Ytr, Xva, Yva, Xte, class_weights):
    """Train seed-ensemble + TopK, return mean probs on val & test."""
    probs_va_list, probs_te_list = [], []
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_fn((Xtr.shape[1], Xtr.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_K)

        model.fit(
            Xtr, Ytr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(Xva, Yva),   # <-- FIXED: Yva اضافه شد
            class_weight=class_weights,
            verbose=0,
            callbacks=[es, rlr, topk]
        )

        topk.set_topk_weights()
        probs_va_list.append(model.predict(Xva, verbose=0).ravel())
        probs_te_list.append(model.predict(Xte, verbose=0).ravel())

    return np.mean(probs_va_list, axis=0), np.mean(probs_te_list, axis=0)

# ======================= GroupKFold =======================
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
picked_weights_all = []
picked_thresholds_all = []

fold = 1
for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)

    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    # فایل‌های train و لیبل فایل‌ها
    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}

    # انتخاب فایل‌های ولیدیشن (file-level stratified)
    val_files = choose_files_for_validation_stratified(train_files, y_file, VAL_FILE_RATIO,
                                                       seed=RANDOM_STATE + fold)
    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # نرمال‌سازی per-fold
    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    # class weights (+NEG_BOOST)
    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # ===== Train base models & get window-level probs =====
    va_cnn, te_cnn = fit_predict_model(build_cnn,          Xtr_in, Ytr_in, Xva_in, Yva_in, Xte, cw)
    va_trf, te_trf = fit_predict_model(lambda s: build_transformer(s, 64, 4, 1, 128, 0.2),
                                       Xtr_in, Ytr_in, Xva_in, Yva_in, Xte, cw)
    va_inc, te_inc = fit_predict_model(lambda s: build_inceptiontime(s, 3, 32, 32, 0.2),
                                       Xtr_in, Ytr_in, Xva_in, Yva_in, Xte, cw)

    # ===== Aggregate to file-level =====
    va_cnn_f = agg_probs_file_level(va_cnn, Fva_in, mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    va_trf_f = agg_probs_file_level(va_trf, Fva_in, mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    va_inc_f = agg_probs_file_level(va_inc, Fva_in, mode=AGG_MODE, trim_q=AGG_TRIM_Q)

    te_cnn_f = agg_probs_file_level(te_cnn, Fte,   mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    te_trf_f = agg_probs_file_level(te_trf, Fte,   mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    te_inc_f = agg_probs_file_level(te_inc, Fte,   mode=AGG_MODE, trim_q=AGG_TRIM_Q)

    # فایل‌سطح ground-truth
    val_true = {int(f): int(Y[f]) for f in va_cnn_f.keys()}
    test_true= {int(f): int(Y[f]) for f in te_cnn_f.keys()}

    # ===== Platt calibration per-model (اختیاری) =====
    def _platt_fit_apply(val_probs_dict, test_probs_dict, y_true_dict):
        files_v = sorted(val_probs_dict.keys())
        v_p = np.array([val_probs_dict[f] for f in files_v], dtype=float)[:, None]
        v_y = np.array([y_true_dict[f] for f in files_v], dtype=int)
        if USE_PLATT_CAL and len(np.unique(v_y))==2:
            calib = LogisticRegression(max_iter=1000)
            calib.fit(v_p, v_y)
            def trans(d):
                files = sorted(d.keys())
                p = np.array([d[f] for f in files], dtype=float)[:, None]
                pc = calib.predict_proba(p)[:,1]
                return {f: float(pc[i]) for i, f in enumerate(files)}
            return trans(val_probs_dict), trans(test_probs_dict)
        else:
            return val_probs_dict, test_probs_dict

    va_cnn_f, te_cnn_f = _platt_fit_apply(va_cnn_f, te_cnn_f, val_true)
    va_trf_f, te_trf_f = _platt_fit_apply(va_trf_f, te_trf_f, val_true)
    va_inc_f, te_inc_f = _platt_fit_apply(va_inc_f, te_inc_f, val_true)

    # ===== Auto-weight selection on validation =====
    best_val_acc, best_w, best_thr = -1.0, None, 0.5
    best_val_prec = 0.0

    for (w_cnn, w_trf, w_inc) in WEIGHT_CANDIDATES:
        blend_val = {f: (w_cnn*va_cnn_f[f] + w_trf*va_trf_f[f] + w_inc*va_inc_f[f]) for f in va_cnn_f.keys()}
        thr, acc, prec = thr_grid_best_acc(val_true, blend_val, THR_MIN, THR_MAX, THR_STEPS)
        if acc > best_val_acc or (acc == best_val_acc and prec > best_val_prec):
            best_val_acc, best_val_prec = acc, prec
            best_thr = thr
            best_w = (w_cnn, w_trf, w_inc)

    picked_weights_all.append(best_w); picked_thresholds_all.append(best_thr)
    print(f"[FOLD {fold}] Picked weights={list(np.round(best_w,3))} | thr={best_thr:.3f} | "
          f"Acc(val)={best_val_acc:.3f}, Prec(val)={best_val_prec:.3f}")

    # ===== Evaluate on test =====
    blend_test = {f: (best_w[0]*te_cnn_f[f] + best_w[1]*te_trf_f[f] + best_w[2]*te_inc_f[f]) for f in te_cnn_f.keys()}
    files_te = sorted(blend_test.keys())
    y_true = np.array([test_true[f] for f in files_te], dtype=int)
    p_vec  = np.array([blend_test[f] for f in files_te], dtype=float)
    y_hat  = (p_vec > best_thr).astype(int)

    acc  = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)
    f1   = f1_score(y_true, y_hat, zero_division=0)

    rows.append({"fold":fold,"acc":acc,"precision":prec,"recall":rec,"f1":f1,
                 "w_cnn":best_w[0],"w_trf":best_w[1],"w_inc":best_w[2],"thr":best_thr})

    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

mean = df[["acc","precision","recall","f1"]].mean()
std  = df[["acc","precision","recall","f1"]].std()

summary = {
    "config": {
        "WIN": WIN, "STR": STR, "Seeds": SEEDS, "TopK": TOPK_K,
        "NEG_BOOST": NEG_BOOST, "AGG_MODE": AGG_MODE, "AGG_TRIM_Q": AGG_TRIM_Q,
        "VAL_FILE_RATIO": VAL_FILE_RATIO, "THR_RANGE": [THR_MIN, THR_MAX, THR_STEPS],
        "USE_PLATT_CAL": USE_PLATT_CAL,
        "Transformer": {"d_model":64,"heads":4,"layers":1,"ffn":128,"dropout":0.2},
        "weight_candidates": WEIGHT_CANDIDATES
    },
    "picked_weights": [tuple(map(float,w)) for w in picked_weights_all],
    "picked_thresholds": [float(t) for t in picked_thresholds_all],
    "Mean±STD": {
        "Accuracy": [float(mean["acc"]), float(std["acc"])],
        "Precision":[float(mean["precision"]), float(std["precision"])],
        "Recall":   [float(mean["recall"]), float(std["recall"])],
        "F1":       [float(mean["f1"]), float(std["f1"])],
    }
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# چاپ جدول قشنگ
table = df.copy()
table["w"] = table.apply(lambda r: f'({r["w_cnn"]:.2f},{r["w_trf"]:.2f},{r["w_inc"]:.2f})', axis=1)
table = table[["fold","acc","precision","recall","f1","thr","w"]]
print("\n[PER-FOLD RESULTS]")
print(table.to_string(index=False, formatters={
    "acc":lambda x:f"{x:.3f}",
    "precision":lambda x:f"{x:.3f}",
    "recall":lambda x:f"{x:.3f}",
    "f1":lambda x:f"{x:.3f}",
    "thr":lambda x:f"{x:.3f}",
}))

print("\n[MEAN ± STD]")
print(f"Accuracy : {mean['acc']:.4f} ± {std['acc']:.4f}")
print(f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}")
print(f"Recall   : {mean['recall']:.4f} ± {std['recall']:.4f}")
print(f"F1       : {mean['f1']:.4f} ± {std['f1']:.4f}")

print("\nSaved to:", RESULT_DIR, flush=True)
