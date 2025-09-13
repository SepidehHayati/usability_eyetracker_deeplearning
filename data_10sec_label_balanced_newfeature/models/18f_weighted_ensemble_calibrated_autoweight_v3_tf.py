# 18f_weighted_ensemble_calibrated_autoweight_v3_tf.py
# هدف: بهبود اکیوریسی با:
#  (1) گرید آستانه‌ی ریزتر (THR_STEPS=201)
#  (2) جست‌وجوی خودکار وزن‌ها اطراف بهترین وزن‌ها (local search ±0.1 با گام 0.02)
#  (3) حفظ همان سه مدل: CNN ساده، Transformer کم‌پارامتر، InceptionTime سبک
#  (4) کالیبراسیون (Platt) روی ولیدیشن و اعمال روی تست
#  (5) چاپ جدول خوانا و ذخیره‌ی full summary

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
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, Input, Add, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "18f_weighted_ensemble_calibrated_autoweight_v3_tf")
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

# گرید آستانه‌ی ریزتر
THR_MIN, THR_MAX = 0.40, 0.80
THR_STEPS        = 201  # قبلاً 41، حالا ریزتر

SEEDS           = [7, 17, 23]
TOPK_K          = 5
NEG_BOOST       = 1.4
AGG_MODE        = "median"   # یا "trimmed"
AGG_TRIM_Q      = 0.10       # اگر trimmed استفاده شود

USE_PLATT_CAL   = True

# کاندیدهای پایه‌ی وزن‌ها (cnn, trf, inc)
BASE_WEIGHT_CANDIDATES = [
    [0.2, 0.5, 0.3],
    [0.25, 0.5, 0.25],
    [0.3, 0.4, 0.3],
    [0.3, 0.5, 0.2],
    [0.4, 0.4, 0.2],
    [0.4, 0.3, 0.3],
    [0.2, 0.4, 0.4],
    [0.33, 0.34, 0.33],
]

# Local search اطراف بهترین وزن‌ها: ±0.1 با گام 0.02
W_DELTA_MAX   = 0.10
W_DELTA_STEP  = 0.02

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
    if 0 in w:
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
            mask = (arr >= lo) & (arr <= hi)
            out[f] = float(np.mean(arr[mask])) if np.any(mask) else float(np.mean(arr))
        else:
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

def search_best_threshold(y_true_files, p_files, tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS):
    grid = np.linspace(tmin, tmax, steps)
    y = np.array([y_true_files[f] for f in p_files.keys()], dtype=int)
    p = np.array([p_files[f] for f in p_files.keys()], dtype=float)
    best_thr, best_acc = 0.5, -1.0
    for t in grid:
        yhat = (p > t).astype(int)
        acc = accuracy_score(y, yhat)
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return best_thr, best_acc

def evaluate_file_level(y_true_files, p_files, thr):
    files = sorted(p_files.keys())
    y = np.array([y_true_files[f] for f in files], dtype=int)
    p = np.array([p_files[f] for f in files], dtype=float)
    yhat = (p > thr).astype(int)
    return {
        "acc": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
    }

def local_weight_neighborhood(w, delta_max=W_DELTA_MAX, step=W_DELTA_STEP):
    # تولید وزن‌های نزدیک به w با تغییرات کوچک، نرمال‌سازی، حذف وزن منفی
    ws = []
    deltas = np.arange(-delta_max, delta_max + 1e-9, step)
    for d0 in deltas:
        for d1 in deltas:
            for d2 in deltas:
                w_new = np.array([w[0]+d0, w[1]+d1, w[2]+d2], dtype=float)
                if np.any(w_new < 0):  # حذف وزن‌های منفی
                    continue
                s = w_new.sum()
                if s <= 0:
                    continue
                w_new = w_new / s
                ws.append(tuple(np.round(w_new, 4)))
    # یکتا
    ws = list(sorted(set(ws)))
    return [list(t) for t in ws]

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

# ساده و سریع: یک بلاک encoder
def build_transformer(input_shape, d_model=64, num_heads=4, ff_dim=128, dropout=0.2):
    inp = Input(shape=input_shape)
    x = inp
    # پروجکشن به d_model
    x = Conv1D(d_model, 1, padding="same")(x)
    # multi-head self-attention
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)
    attn_out = attn(x, x)
    x = Add()([x, attn_out])
    x = LayerNorm()(x)

    # FFN
    f = Sequential([Dense(ff_dim, activation='relu'),
                    Dropout(dropout),
                    Dense(d_model)])(x)
    x = Add()([x, f])
    x = LayerNorm()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(DROPOUT_HEAD)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer="ones")
        self.beta  = self.add_weight(name="beta",  shape=input_shape[-1:], initializer="zeros")
    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var  = tf.reduce_mean(tf.square(x-mean), axis=-1, keepdims=True)
        return self.gamma * (x-mean) / tf.sqrt(var + self.eps) + self.beta

# InceptionTime سبک: 3 ماژول
def inception_module(x, nf=32, bottleneck=32, kernel_sizes=(9,19,39), drop=0.2):
    if bottleneck and x.shape[-1] > 1:
        x = Conv1D(bottleneck, 1, padding='same', use_bias=False)(x)
    c1 = Conv1D(nf, kernel_sizes[0], padding='same', use_bias=False)(x)
    c2 = Conv1D(nf, kernel_sizes[1], padding='same', use_bias=False)(x)
    c3 = Conv1D(nf, kernel_sizes[2], padding='same', use_bias=False)(x)
    p  = tf.keras.layers.MaxPool1D(3, strides=1, padding='same')(x)
    p  = Conv1D(nf, 1, padding='same', use_bias=False)(p)
    x  = Concatenate()([c1,c2,c3,p])
    x  = BatchNormalization()(x)
    x  = Activation('relu')(x)
    x  = Dropout(drop)(x)
    return x

def build_inception(input_shape, modules=3, nf=32, bottleneck=32, drop=0.2):
    inp = Input(shape=input_shape)
    x = inp
    for _ in range(modules):
        x = inception_module(x, nf=nf, bottleneck=bottleneck, drop=drop)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(DROPOUT_HEAD)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======================= Train helpers =======================
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

def fit_predict_model(build_fn, Xtr, Ytr, Xva, Xte, cw):
    val_ps, tst_ps = [], []
    for sd in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(sd); tf.random.set_seed(sd)
        model = build_fn((Xtr.shape[1], Xtr.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_K)
        model.fit(Xtr, Ytr,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(Xva, Yva),
                  class_weight=cw,
                  verbose=0,
                  callbacks=[es, rlr, topk])
        topk.set_topk_weights()
        val_ps.append(model.predict(Xva, verbose=0).ravel())
        tst_ps.append(model.predict(Xte, verbose=0).ravel())
    val_p = np.mean(val_ps, axis=0)
    tst_p = np.mean(tst_ps, axis=0)
    return val_p, tst_p

# ======================= Main CV loop =======================
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
picked_weights_all = []
picked_thrs_all = []

fold = 1
for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)

    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}

    val_files = choose_files_for_validation_stratified(train_files, y_file, VAL_FILE_RATIO,
                                                       seed=RANDOM_STATE + fold)
    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva,   Fva_in  = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # استانداردسازی
    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    # وزن کلاس‌ها
    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # ===== Train 3 models and get window-level probs
    # CNN
    va_win_cnn, te_win_cnn = fit_predict_model(build_cnn,          Xtr_in, Ytr_in, Xva_in, Xte, cw)
    # Transformer
    va_win_trf, te_win_trf = fit_predict_model(
        lambda shp: build_transformer(shp, d_model=64, num_heads=4, ff_dim=128, dropout=0.2),
        Xtr_in, Ytr_in, Xva_in, Xte, cw
    )
    # InceptionTime
    va_win_inc, te_win_inc = fit_predict_model(
        lambda shp: build_inception(shp, modules=3, nf=32, bottleneck=32, drop=0.2),
        Xtr_in, Ytr_in, Xva_in, Xte, cw
    )

    # ===== Aggregate to file level
    va_file_cnn = agg_probs_file_level(va_win_cnn, Fva_in, mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    te_file_cnn = agg_probs_file_level(te_win_cnn, Fte,    mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    va_file_trf = agg_probs_file_level(va_win_trf, Fva_in, mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    te_file_trf = agg_probs_file_level(te_win_trf, Fte,    mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    va_file_inc = agg_probs_file_level(va_win_inc, Fva_in, mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    te_file_inc = agg_probs_file_level(te_win_inc, Fte,    mode=AGG_MODE, trim_q=AGG_TRIM_Q)

    # ===== Platt calibration per model (fit on val, apply to test)
    def platt_fit_apply(va_dict, te_dict):
        files_v = sorted(va_dict.keys())
        pv = np.array([va_dict[f] for f in files_v])[:, None]
        yv = np.array([Y[f] for f in files_v], dtype=int)
        if len(np.unique(yv)) < 2 or not USE_PLATT_CAL:
            # کلا بدون کالیبراسیون
            return va_dict, te_dict
        calib = LogisticRegression(max_iter=1000)
        calib.fit(pv, yv)
        files_t = sorted(te_dict.keys())
        pt = np.array([te_dict[f] for f in files_t])[:, None]
        va_new = {f: float(calib.predict_proba(np.array([[va_dict[f]]]))[0,1]) for f in files_v}
        te_new = {f: float(p) for f, p in zip(files_t, calib.predict_proba(pt)[:,1])}
        return va_new, te_new

    va_file_cnn, te_file_cnn = platt_fit_apply(va_file_cnn, te_file_cnn)
    va_file_trf, te_file_trf = platt_fit_apply(va_file_trf, te_file_trf)
    va_file_inc, te_file_inc = platt_fit_apply(va_file_inc, te_file_inc)

    # ===== Compose validation dicts for weight search
    y_true_val_files = {int(f): int(Y[f]) for f in va_file_cnn.keys()}  # همه یکسانند
    y_true_te_files  = {int(f): int(Y[f]) for f in te_file_cnn.keys()}

    # تابع کمکی برای ترکیب سه مدل با وزن‌ها
    def combine_with_weights(w, d1, d2, d3):
        files = d1.keys()
        return {int(f): float(w[0]*d1[f] + w[1]*d2[f] + w[2]*d3[f]) for f in files}

    # ابتدا روی کاندیدهای پایه جست‌وجو می‌کنیم
    best = {"weights": None, "thr": 0.5, "val_acc": -1.0}
    for w in BASE_WEIGHT_CANDIDATES:
        p_val = combine_with_weights(w, va_file_cnn, va_file_trf, va_file_inc)
        thr, val_acc = search_best_threshold(y_true_val_files, p_val, THR_MIN, THR_MAX, THR_STEPS)
        if val_acc > best["val_acc"]:
            best = {"weights": w, "thr": float(thr), "val_acc": float(val_acc)}

    # سپس local search اطراف بهترین وزن‌ها
    neighbors = local_weight_neighborhood(best["weights"], W_DELTA_MAX, W_DELTA_STEP)
    for w in neighbors:
        p_val = combine_with_weights(w, va_file_cnn, va_file_trf, va_file_inc)
        thr, val_acc = search_best_threshold(y_true_val_files, p_val, THR_MIN, THR_MAX, THR_STEPS)
        if val_acc > best["val_acc"]:
            best = {"weights": w, "thr": float(thr), "val_acc": float(val_acc)}

    print(f"[FOLD {fold}] Picked weights={best['weights']} | thr={best['thr']:.3f} | Acc(val)={best['val_acc']:.3f}")

    # اعمال روی تست
    p_te = combine_with_weights(best["weights"], te_file_cnn, te_file_trf, te_file_inc)
    metrics = evaluate_file_level(y_true_te_files, p_te, best["thr"])

    rows.append({
        "fold": fold,
        "acc": metrics["acc"], "precision": metrics["precision"],
        "recall": metrics["recall"], "f1": metrics["f1"],
        "thr": best["thr"], "w_cnn": best["weights"][0], "w_trf": best["weights"][1], "w_inc": best["weights"][2]
    })
    picked_weights_all.append(best["weights"])
    picked_thrs_all.append(best["thr"])
    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)

# چاپ جدول خوانا
print("\n[PER-FOLD RESULTS]")
print(f"{'fold':>4} {'acc':>6} {'precision':>10} {'recall':>7} {'f1':>6} {'thr':>6}    weights(cnn,trf,inc)")
for _, r in df.iterrows():
    print(f"{int(r['fold']):4d} {r['acc']:.3f} {r['precision']:.3f} {r['recall']:.3f} {r['f1']:.3f} {r['thr']:.3f}    ({r['w_cnn']:.2f},{r['w_trf']:.2f},{r['w_inc']:.2f})")

print("\n[MEAN ± STD]")
print(f"Accuracy : {mean['acc']:.4f} ± {std['acc']:.4f}")
print(f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}")
print(f"Recall   : {mean['recall']:.4f} ± {std['recall']:.4f}")
print(f"F1       : {mean['f1']:.4f} ± {std['f1']:.4f}")

summary = {
    "config": {
        "WIN": WIN, "STR": STR, "Seeds": SEEDS, "TopK": TOPK_K,
        "NEG_BOOST": NEG_BOOST, "AGG_MODE": AGG_MODE, "AGG_TRIM_Q": AGG_TRIM_Q,
        "VAL_FILE_RATIO": VAL_FILE_RATIO,
        "THR_RANGE": [THR_MIN, THR_MAX, THR_STEPS],
        "USE_PLATT_CAL": USE_PLATT_CAL,
        "Transformer": {"d_model": 64, "heads": 4, "layers": 1, "ffn": 128, "dropout": 0.2},
        "weight_candidates_base": BASE_WEIGHT_CANDIDATES,
        "local_weight_search": {"delta_max": W_DELTA_MAX, "step": W_DELTA_STEP}
    },
    "picked_weights": picked_weights_all,
    "picked_thresholds": picked_thrs_all,
    "Mean±STD": {
        "Accuracy":  [float(mean['acc']), float(std['acc'])],
        "Precision": [float(mean['precision']), float(std['precision'])],
        "Recall":    [float(mean['recall']), float(std['recall'])],
        "F1":        [float(mean['f1']), float(std['f1'])],
    }
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to: {RESULT_DIR}", flush=True)
