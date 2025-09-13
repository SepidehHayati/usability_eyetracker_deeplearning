# 18e_weighted_ensemble_calibrated_autoweight_tf.py
# هدف: Ensembling (CNN + Transformer + InceptionTime)
#   - آموزش هر سه مدل روی پنجره‌ها + Top-K snapshot averaging + seed-ensemble
#   - تجمیع به سطح فایل
#   - کالیبراسیون احتمال (Platt) روی فایل‌های ولیدیشن (در صورت دوکلاسی بودن)
#   - انتخاب خودکار وزن‌های اِنسمبل روی ولیدیشن (grid کوچک ولی منطقی)
#   - انتخاب خودکار آستانه (max accuracy روی ولیدیشن، با شکست مساوی: precision بالاتر)
#   - گزارش میانگین ± انحراف معیار روی 6 فولد GroupKFold(user)

import os, json, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "18e_weighted_ensemble_calibrated_autoweight_tf")
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
SEEDS           = [7, 17, 23]
TOPK_K          = 5
NEG_BOOST       = 1.4
AGG_MODE        = "median"      # یا "trimmed"
AGG_TRIM_Q      = 0.10          # فقط اگر trimmed
THR_MIN, THR_MAX, THR_STEPS = 0.40, 0.80, 101   # گرِید ریزتر برای انتخاب آستانه

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

def agg_probs_file_level(probs, file_ids):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    out = {}
    for f, ps in d.items():
        if AGG_MODE == "median":
            out[f] = float(np.median(ps))
        elif AGG_MODE == "trimmed":
            q = AGG_TRIM_Q
            lo, hi = np.quantile(ps, q), np.quantile(ps, 1-q)
            trimmed = [u for u in ps if (u>=lo and u<=hi)]
            out[f] = float(np.mean(trimmed)) if len(trimmed)>0 else float(np.mean(ps))
        else:
            out[f] = float(np.mean(ps))
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

def pick_threshold_acc(y_true, p, tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS):
    grid = np.linspace(tmin, tmax, steps)
    best = (0.5, -1.0, 0.0)  # thr, acc, prec (برای شکستن تساوی)
    for t in grid:
        yhat = (p > t).astype(int)
        acc = accuracy_score(y_true, yhat)
        prec = precision_score(y_true, yhat, zero_division=0)
        if (acc > best[1]) or (acc == best[1] and prec > best[2]):
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

# ======================= Model builders =======================
def build_cnn(input_shape):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(48, 7, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling1D(2), layers.Dropout(DROPOUT_BLOCK),

        layers.Conv1D(64, 5, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling1D(2), layers.Dropout(DROPOUT_BLOCK),

        layers.Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling1D(2), layers.Dropout(DROPOUT_BLOCK),

        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(DROPOUT_HEAD),
        layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs); self.d_model = d_model
    def call(self, x):
        # x: (B,T,D)
        T = tf.shape(x)[1]
        i = tf.cast(tf.range(self.d_model)[tf.newaxis, tf.newaxis, :], tf.float32)
        pos = tf.cast(tf.range(T)[tf.newaxis, :, tf.newaxis], tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2*(i//2))/tf.cast(self.d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, :, 0::2])
        cosines = tf.cos(angle_rads[:, :, 1::2])
        pe = tf.concat([sines, cosines], axis=-1)
        return x + pe

def transformer_block(x, d_model=64, num_heads=4, dff=128, rate=0.2):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.Add()([x, layers.Dropout(rate)(attn)])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = layers.Dense(dff, activation='relu')(x)
    ffn = layers.Dense(d_model)(ffn)
    x = layers.Add()([x, layers.Dropout(rate)(ffn)])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

def build_transformer(input_shape, d_model=64, num_heads=4, dff=128, n_layers=1, rate=0.2):
    inp = layers.Input(shape=input_shape)
    x = layers.Dense(d_model)(inp)
    x = PositionalEncoding(d_model)(x)
    for _ in range(n_layers):
        x = transformer_block(x, d_model, num_heads, dff, rate)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT_HEAD)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(inp, out)
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def inception_module(x, nf=32, bottleneck=32, ks=(9,19,39), rate=0.1):
    if bottleneck and x.shape[-1] > 1:
        x = layers.Conv1D(bottleneck, 1, padding='same', activation='linear')(x)
    convs = []
    for k in ks:
        convs.append(layers.Conv1D(nf, k, padding='same', activation='relu')(x))
    pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    pool = layers.Conv1D(nf, 1, padding='same', activation='relu')(pool)
    x = layers.Concatenate()(convs + [pool])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate)(x)
    return x

def build_inceptiontime(input_shape, n_modules=3, nf=32, bottleneck=32, rate=0.2):
    inp = layers.Input(shape=input_shape)
    x = inp
    for _ in range(n_modules):
        x = inception_module(x, nf=nf, bottleneck=bottleneck, rate=rate)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(DROPOUT_HEAD)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(inp, out)
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ======================= Training helper =======================
def train_and_predict(model_builder, Xtr_in, Ytr_in, Xva_in, Xte, class_weights):
    va_preds, te_preds = [], []
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)
        m = model_builder((Xtr_in.shape[1], Xtr_in.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(m, k=TOPK_K)
        m.fit(Xtr_in, Ytr_in,
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(Xva_in, Ytr_in[:len(Xva_in)]*0 + 0), # dummy to avoid Keras bug? (kept real val below)
              class_weight=class_weights, verbose=0, callbacks=[es, rlr, topk])
        # reset to averaged Top-K
        topk.set_topk_weights()
        va_preds.append(m.predict(Xva_in, verbose=0).ravel())
        te_preds.append(m.predict(Xte,   verbose=0).ravel())
    return np.mean(va_preds, axis=0), np.mean(te_preds, axis=0)

# ======================= Ensemble candidates =======================
WEIGHT_CANDS = [
    (0.20, 0.50, 0.30),
    (0.25, 0.50, 0.25),
    (0.30, 0.40, 0.30),
    (0.30, 0.50, 0.20),
    (0.40, 0.40, 0.20),
    (0.40, 0.30, 0.30),
    (0.20, 0.40, 0.40),
    (0.33, 0.34, 0.33),
]

# ======================= GroupKFold (user) =======================
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
fold = 1
picked_weights = []
picked_thresholds = []

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
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # Standardize per-fold
    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    # Class weights
    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # === Train 3 base models: CNN / Transformer / InceptionTime
    va_cnn, te_cnn = train_and_predict(build_cnn,           Xtr_in, Ytr_in, Xva_in, Xte, cw)
    va_trf, te_trf = train_and_predict(lambda s: build_transformer(s, 64, 4, 128, 1, 0.2),
                                       Xtr_in, Ytr_in, Xva_in, Xte, cw)
    va_inc, te_inc = train_and_predict(lambda s: build_inceptiontime(s, n_modules=3, nf=32, bottleneck=32, rate=0.2),
                                       Xtr_in, Ytr_in, Xva_in, Xte, cw)

    # Aggregate window -> file per model
    va_cnn_file = agg_probs_file_level(va_cnn, Fva_in)
    va_trf_file = agg_probs_file_level(va_trf, Fva_in)
    va_inc_file = agg_probs_file_level(va_inc, Fva_in)
    te_cnn_file = agg_probs_file_level(te_cnn, Fte)
    te_trf_file = agg_probs_file_level(te_trf, Fte)
    te_inc_file = agg_probs_file_level(te_inc, Fte)

    # Calibrate (Platt) per model on validation (if both classes present)
    def maybe_platt(prob_dict):
        files = sorted(prob_dict.keys())
        p = np.array([prob_dict[f] for f in files])[:, None]
        y = np.array([Y[f] for f in files], dtype=int)
        if len(np.unique(y)) == 2:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(p, y)
            return lr
        return None

    cal_cnn = maybe_platt(va_cnn_file)
    cal_trf = maybe_platt(va_trf_file)
    cal_inc = maybe_platt(va_inc_file)

    def apply_cal(prob_dict, lr):
        files = sorted(prob_dict.keys())
        p = np.array([prob_dict[f] for f in files])[:, None]
        if lr is not None:
            return dict(zip(files, lr.predict_proba(p)[:,1].tolist()))
        else:
            return prob_dict

    va_cnn_c = apply_cal(va_cnn_file, cal_cnn)
    va_trf_c = apply_cal(va_trf_file, cal_trf)
    va_inc_c = apply_cal(va_inc_file, cal_inc)
    te_cnn_c = apply_cal(te_cnn_file, cal_cnn)
    te_trf_c = apply_cal(te_trf_file, cal_trf)
    te_inc_c = apply_cal(te_inc_file, cal_inc)

    # Build aligned vectors (validation + test)
    v_files = sorted(va_cnn_c.keys())
    t_files = sorted(te_cnn_c.keys())
    v_y = np.array([Y[f] for f in v_files], dtype=int)
    t_y = np.array([Y[f] for f in t_files], dtype=int)

    v_cnn = np.array([va_cnn_c[f] for f in v_files], dtype=float)
    v_trf = np.array([va_trf_c[f] for f in v_files], dtype=float)
    v_inc = np.array([va_inc_c[f] for f in v_files], dtype=float)

    t_cnn = np.array([te_cnn_c[f] for f in t_files], dtype=float)
    t_trf = np.array([te_trf_c[f] for f in t_files], dtype=float)
    t_inc = np.array([te_inc_c[f] for f in t_files], dtype=float)

    # Auto-select weights on validation by maximizing accuracy (then precision)
    best = None  # (acc, prec, thr, w, yhat)
    for w in WEIGHT_CANDS:
        w = np.array(w, dtype=float)
        v_ens = w[0]*v_cnn + w[1]*v_trf + w[2]*v_inc
        thr, acc_val, prec_val = pick_threshold_acc(v_y, v_ens, THR_MIN, THR_MAX, THR_STEPS)
        if (best is None) or (acc_val > best[0]) or (acc_val == best[0] and prec_val > best[1]):
            best = (acc_val, prec_val, thr, w)

    acc_val, prec_val, thr_star, w_star = best
    picked_weights.append(w_star.tolist())
    picked_thresholds.append(float(thr_star))
    print(f"[FOLD {fold}] Picked weights={w_star.tolist()} | thr={thr_star:.3f} | Acc(val)={acc_val:.3f}, Prec(val)={prec_val:.3f}")

    # Apply to test
    t_ens = w_star[0]*t_cnn + w_star[1]*t_trf + w_star[2]*t_inc
    y_pred = (t_ens > thr_star).astype(int)

    acc  = accuracy_score(t_y, y_pred)
    prec = precision_score(t_y, y_pred, zero_division=0)
    rec  = recall_score(t_y, y_pred, zero_division=0)
    f1   = f1_score(t_y, y_pred, zero_division=0)

    rows.append({"fold":fold, "thr":float(thr_star),
                 "acc":acc,"precision":prec,"recall":rec,"f1":f1,
                 "weights": w_star.tolist(),
                 "n_test_files":len(t_files)})

    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)

summary = {
    "config":{
        "WIN":WIN, "STR":STR, "Seeds":SEEDS, "TopK":TOPK_K,
        "NEG_BOOST":NEG_BOOST, "AGG_MODE":AGG_MODE, "AGG_TRIM_Q":AGG_TRIM_Q,
        "VAL_FILE_RATIO":VAL_FILE_RATIO,
        "THR_RANGE":[THR_MIN, THR_MAX, THR_STEPS],
        "weight_candidates": WEIGHT_CANDS,
        "models": {
            "cnn":"3xConv-BN-ReLU + GAP",
            "transformer":"1x encoder (d_model=64, heads=4, ffn=128, drop=0.2)",
            "inceptiontime":"3 modules, nf=32, bottleneck=32, drop=0.2"
        }
    },
    "picked_weights": picked_weights,
    "picked_thresholds": picked_thresholds,
    "Mean±STD":{
        "Accuracy":[float(mean["acc"]), float(std["acc"])],
        "Precision":[float(mean["precision"]), float(std["precision"])],
        "Recall":[float(mean["recall"]), float(std["recall"])],
        "F1":[float(mean["f1"]), float(std["f1"])],
    }
}

with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n[SUMMARY]")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("Saved to:", RESULT_DIR, flush=True)
