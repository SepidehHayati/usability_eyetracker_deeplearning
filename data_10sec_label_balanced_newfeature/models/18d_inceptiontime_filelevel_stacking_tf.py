# 18d_inceptiontime_filelevel_stacking_tf.py
# هدف: فقط InceptionTime را روی پنجره‌ها اجرا می‌کنیم، احتمال‌ها را به سطح فایل تجمیع
# می‌کنیم و روی یک سری متافچر ساده فایل‌سطح (mean/median/max/.../frac_gt_thr/IQR و غیره)
# یک Logistic Regression سبک برای طبقه‌بندی فایل‌سطح می‌سازیم.
# پیاده‌سازی سازگار با ساختار پروژه شما (X8.npy / Y8.npy / G8.npy).

import os, json, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation,
                                     Concatenate, GlobalAveragePooling1D, Dropout, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "18d_inceptiontime_filelevel_stacking_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ======================= Config =======================
WIN, STR         = 500, 200
N_SPLITS         = 6
SEEDS            = [7, 17, 23]
EPOCHS           = 80
BATCH_SIZE       = 16
LR_INIT          = 1e-3
L2_REG           = 1e-5
DROP_RATE        = 0.2
VAL_FILE_RATIO   = 0.35
THR_GRID         = (0.40, 0.80, 41)  # min, max, steps برای ThrOpt در فایل‌سطح
NEG_BOOST        = 1.4               # وزن بیشتر کلاس 0 برای کاهش FP در آموزش پنجره‌ها
THR_FRAC         = 0.50              # برای متافچر fraction of windows > THR_FRAC

np.random.seed(42); random.seed(42); tf.random.set_seed(42)

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

def agg_probs_by_file(probs, file_ids):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    return {f: np.array(ps, dtype=float) for f, ps in d.items()}

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

def thr_opt_for_acc(p_true_dict, thr_min, thr_max, steps):
    files = sorted(p_true_dict.keys())
    y_true = np.array([p_true_dict[f][0] for f in files], dtype=int)  # label در idx 0
    p_mean = np.array([p_true_dict[f][1] for f in files], dtype=float) # mean prob در idx 1
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(thr_min, thr_max, steps):
        y_hat = (p_mean > t).astype(int)
        acc = accuracy_score(y_true, y_hat)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc

def file_meta_features(prob_vec, thr=THR_FRAC):
    v = np.asarray(prob_vec, dtype=float)
    if v.size == 0:
        return np.zeros(12, dtype=float)
    q1, q3 = np.quantile(v, [0.25, 0.75])
    iqr = q3 - q1
    return np.array([
        v.mean(),                 # 0
        np.median(v),             # 1
        v.max(),                  # 2
        v.min(),                  # 3
        v.var(ddof=0),            # 4
        v.std(ddof=0),            # 5
        (v > thr).mean(),         # 6 fraction > thr
        q1,                       # 7
        q3,                       # 8
        iqr,                      # 9
        (v>0.6).mean(),           # 10 fraction > 0.6
        (v>0.7).mean(),           # 11 fraction > 0.7
    ], dtype=float)

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

# ======================= InceptionTime model =======================
def inception_module(x, nb_filters=32, kernel_sizes=(9,19,39), bottleneck_size=32, stride=1, use_bias=False):
    if bottleneck_size and int(x.shape[-1]) > 1:
        x_in = Conv1D(bottleneck_size, 1, padding='same', use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
        x_in = BatchNormalization()(x_in)
        x_in = Activation('relu')(x_in)
    else:
        x_in = x

    conv_list = []
    for k in kernel_sizes:
        z = Conv1D(nb_filters, kernel_size=k, strides=stride, padding='same',
                   use_bias=use_bias, kernel_regularizer=regularizers.l2(L2_REG))(x_in)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
        conv_list.append(z)

    pool = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(x)
    pool = Conv1D(nb_filters, 1, padding='same', use_bias=use_bias,
                  kernel_regularizer=regularizers.l2(L2_REG))(pool)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    conv_list.append(pool)

    x_out = Concatenate(axis=-1)(conv_list)
    x_out = BatchNormalization()(x_out)
    x_out = Activation('relu')(x_out)
    return x_out

def build_inceptiontime(input_shape, n_modules=3, nb_filters=32, bottleneck=32, drop=DROP_RATE):
    inp = Input(shape=input_shape)
    x = inp
    for _ in range(n_modules):
        x = inception_module(x, nb_filters=nb_filters, bottleneck_size=bottleneck)
    x = GlobalAveragePooling1D()(x)
    if drop > 0:
        x = Dropout(drop)(x)
    x = Dense(64, activation='relu')(x)
    if drop > 0:
        x = Dropout(drop)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======================= Main CV Loop =======================
gkf = GroupKFold(n_splits=N_SPLITS)
rows = []
fold = 1

thr_min, thr_max, thr_steps = THR_GRID

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)
    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}
    val_files = choose_files_for_validation_stratified(train_files, y_file, VAL_FILE_RATIO,
                                                       seed=1000 + fold)

    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # Seed ensemble + TopK snapshot avg
    va_probs_seeds, te_probs_seeds = [], []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_inceptiontime((Xtr_in.shape[1], Xtr_in.shape[2]),
                                    n_modules=3, nb_filters=32, bottleneck=32, drop=DROP_RATE)
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6,
                                min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=5)

        model.fit(Xtr_in, Ytr_in,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(Xva_in, Yva_in),
                  class_weight=cw,
                  verbose=0,
                  callbacks=[es, rlr, topk])

        topk.set_topk_weights()

        va_probs_seeds.append(model.predict(Xva_in, verbose=0).ravel())
        te_probs_seeds.append(model.predict(Xte,   verbose=0).ravel())

    va_probs = np.mean(va_probs_seeds, axis=0)
    te_probs = np.mean(te_probs_seeds, axis=0)

    # تجمیع پنجره → فایل
    va_file_probs = agg_probs_by_file(va_probs, Fva_in)
    te_file_probs = agg_probs_by_file(te_probs, Fte)

    # ساخت متافچرهای فایل‌سطح
    def build_meta_for(files_dict):
        feats = {}
        for f, vec in files_dict.items():
            feats[int(f)] = file_meta_features(vec, thr=THR_FRAC)
        return feats

    va_meta = build_meta_for(va_file_probs)
    te_meta = build_meta_for(te_file_probs)

    # ماتریس‌های X_meta, y_file
    va_files_sorted = sorted(va_meta.keys())
    te_files_sorted = sorted(te_meta.keys())

    X_va_meta = np.stack([va_meta[f] for f in va_files_sorted], axis=0)
    y_va = np.array([Y[f] for f in va_files_sorted], dtype=int)

    X_te_meta = np.stack([te_meta[f] for f in te_files_sorted], axis=0)
    y_te = np.array([Y[f] for f in te_files_sorted], dtype=int)

    # نرمال‌سازی متافچرها (روی ولیدیشن fit می‌کنیم تا leakage نداشته باشه)
    sc_meta = StandardScaler().fit(X_va_meta)
    X_va_meta_sc = sc_meta.transform(X_va_meta)
    X_te_meta_sc = sc_meta.transform(X_te_meta)

    # LR فایل‌سطح
    meta_lr = LogisticRegression(max_iter=2000)
    meta_lr.fit(X_va_meta_sc, y_va)

    # ThrOpt برای بیشینه‌سازی Accuracy روی ولیدیشن
    # ابتدا احتمال فایل‌سطح از LR بگیریم
    va_file_prob_lr = meta_lr.predict_proba(X_va_meta_sc)[:,1]
    te_file_prob_lr = meta_lr.predict_proba(X_te_meta_sc)[:,1]

    p_true_dict_val = {int(f): (int(Y[f]), float(va_file_prob_lr[i]))
                       for i, f in enumerate(va_files_sorted)}
    best_thr, best_acc = thr_opt_for_acc(p_true_dict_val,
                                         thr_min=thr_min, thr_max=thr_max, steps=thr_steps)

    # ارزیابی روی تست
    y_pred = (te_file_prob_lr > best_thr).astype(int)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)

    rows.append({
        "fold": fold,
        "thr": float(best_thr),
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "n_test_files": len(te_files_sorted)
    })

    print(f"[FOLD {fold}] Meta-LR ThrOpt: thr={best_thr:.3f} (val Acc={best_acc:.3f}) | "
          f"Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}", flush=True)

    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_filelevel_meta_lr.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = {
    "config": {
        "WIN": WIN, "STR": STR, "Seeds": SEEDS, "TopK": 5,
        "VAL_FILE_RATIO": VAL_FILE_RATIO,
        "NEG_BOOST": NEG_BOOST,
        "THR_GRID": list(THR_GRID),
        "THR_FRAC": THR_FRAC,
        "model": "InceptionTime (3 modules, nf=32, bottleneck=32, drop=0.2)",
        "meta_features": [
            "mean", "median", "max", "min", "var", "std",
            "frac>0.5", "q1", "q3", "iqr", "frac>0.6", "frac>0.7"
        ]
    },
    "Mean±STD": {
        "Accuracy": [float(mean["acc"]), float(std["acc"])],
        "Precision": [float(mean["precision"]), float(std["precision"])],
        "Recall": [float(mean["recall"]), float(std["recall"])],
        "F1": [float(mean["f1"]), float(std["f1"])],
    },
    "thr_list": [float(t) for t in df["thr"].tolist()]
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n[SUMMARY]")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("Saved to:", RESULT_DIR)
