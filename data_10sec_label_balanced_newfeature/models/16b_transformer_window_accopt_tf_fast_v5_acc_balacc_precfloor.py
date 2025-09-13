# 16b_transformer_window_accopt_tf_fast_v5_acc_balacc_precfloor.py
# هدف: بهبود Accuracy فایل‌سطح با انتخاب آستانه بهینه:
# 1) بیشینه‌سازی Accuracy با قید Precision و Predicted Positive Rate
# 2) در صورت عدم احراز قید، fallback: بیشینه‌سازی Balanced Accuracy
# 3) در صورت عدم امکان، fallback: thr=0.50
# + کالیبراسیون احتمال (Platt) روی فایل‌های ولیدیشن در هر فولد، اگر هر دو کلاس حاضر باشند.

import os, math, random, json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score
)

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "16b_transformer_window_accopt_tf_fast_v5_acc_balacc_precfloor")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ======================= Config =======================
WIN, STR        = 500, 200         # پنجره ۱۰ ثانیه‌ای → 500 نمونه با فرض ~50 Hz (اینجا طبق داده شما 1500~10s بود؛ پنجره 1/3 انتخاب شد)
N_SPLITS        = 6
EPOCHS          = 80
BATCH_SIZE      = 16
LR_INIT         = 1e-3
L2_REG          = 1e-5
DROPOUT         = 0.2

RANDOM_STATE    = 42
SEEDS           = [7, 17, 23]
TOPK_K          = 5
VAL_FILE_RATIO  = 0.35

# Aggregation تنظیمات
AGG_MODE        = "median"         # "median" | "mean" | "trimmed"
AGG_TRIM_Q      = 0.10             # برای trimmed-mean (برش 10% دو سر)

# Class-weight boost برای کلاس 0 (non-easy)
NEG_BOOST       = 1.4

# Threshold search grid
THR_MIN, THR_MAX = 0.45, 0.80
THR_STEPS       = 41

# قیود انتخاب آستانه
PREC_FLOOR      = 0.65             # حداقل Precision
PPR_MIN, PPR_MAX = 0.35, 0.65      # دامنه نسبت پیش‌بینی مثبت‌ها

# Transformer تنظیمات
D_MODEL         = 64               # embedding dim
N_HEADS         = 4
N_LAYERS        = 1
FF_DIM          = 128

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)
tf.keras.backend.clear_session()

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
    for f, lst in d.items():
        arr = np.array(lst, dtype=float)
        if mode == "median":
            out[f] = float(np.median(arr))
        elif mode == "trimmed":
            qlo = np.quantile(arr, trim_q)
            qhi = np.quantile(arr, 1.0-trim_q)
            clipped = arr[(arr >= qlo) & (arr <= qhi)]
            out[f] = float(np.mean(clipped)) if len(clipped)>0 else float(np.mean(arr))
        else:  # mean
            out[f] = float(np.mean(arr))
    return out

def choose_files_for_validation_stratified(train_file_ids, y_file_dict, ratio, seed=0):
    rng = np.random.default_rng(seed)
    files0 = [f for f in train_file_ids if y_file_dict[f]==0]
    files1 = [f for f in train_file_ids if y_file_dict[f]==1]
    n_val = max(2, int(math.ceil(len(train_file_ids)*ratio)))  # حداقل 2 فایل
    # سعی کن از هر کلاس حداقل 1 بیاد
    n1 = max(1, int(round(n_val * (len(files1)/(len(files0)+len(files1)+1e-9)))))
    n0 = max(1, n_val - n1)
    val = set()
    if files0:
        val.update(rng.choice(files0, size=min(n0, len(files0)), replace=False).tolist())
    if files1:
        val.update(rng.choice(files1, size=min(n1, len(files1)), replace=False).tolist())
    # اگر هنوز کمتر است، از باقی مانده پر کن
    remaining = [f for f in train_file_ids if f not in val]
    rng.shuffle(remaining)
    while len(val) < n_val and remaining:
        val.add(remaining.pop())
    return val

def threshold_grid():
    return np.linspace(THR_MIN, THR_MAX, THR_STEPS)

def pick_threshold_with_constraints(val_file_probs, val_file_true,
                                    prec_floor=PREC_FLOOR,
                                    ppr_min=PPR_MIN, ppr_max=PPR_MAX):
    files = list(val_file_probs.keys())
    y_true = np.array([val_file_true[f] for f in files], dtype=int)
    p_vec  = np.array([val_file_probs[f] for f in files], dtype=float)
    best = None
    for t in threshold_grid():
        y_hat = (p_vec > t).astype(int)
        prec = precision_score(y_true, y_hat, zero_division=0)
        ppr  = float(np.mean(y_hat==1))
        if prec >= prec_floor and ppr_min <= ppr <= ppr_max:
            acc = accuracy_score(y_true, y_hat)
            # معیار اصلی: Accuracy، بَعداً برای هم‌ارزش‌ها Prec بزرگ‌تر
            key = (acc, prec)
            if (best is None) or (key > best[0]):
                best = (key, float(t), acc, prec, ppr)
    if best is None:
        return None
    _key, t, acc, prec, ppr = best
    return {"thr": t, "acc": acc, "prec": prec, "ppr": ppr, "mode": "acc_constrained"}

def pick_threshold_balacc(val_file_probs, val_file_true):
    files = list(val_file_probs.keys())
    y_true = np.array([val_file_true[f] for f in files], dtype=int)
    p_vec  = np.array([val_file_probs[f] for f in files], dtype=float)
    best = None
    for t in threshold_grid():
        y_hat = (p_vec > t).astype(int)
        bal = balanced_accuracy_score(y_true, y_hat)
        if (best is None) or (bal > best[0]):
            best = (bal, float(t))
    bal, t = best
    return {"thr": t, "balacc": bal, "mode": "balanced_accuracy"}

# ======================= Top-K Saver =======================
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

# ======================= Transformer Model =======================
def transformer_block(x, d_model, num_heads, ff_dim, dropout=0.0):
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_out = layers.Dropout(dropout)(attn_out)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    ffn = layers.Dense(ff_dim, activation="relu")(out1)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2

def build_model(input_shape):
    inp = layers.Input(shape=input_shape)      # (T_win, C)
    # خطی‌سازی کانال‌ها به d_model
    x = layers.Dense(D_MODEL)(inp)
    # پوزیشنال اِنکودینگ سینوسی
    positions = tf.range(start=0, limit=input_shape[0], delta=1, dtype=tf.float32)  # (T_win,)
    pos_enc = get_sinusoidal_positional_encoding(input_shape[0], D_MODEL)           # (T_win, d_model)
    pos_enc = tf.cast(pos_enc, tf.float32)
    x = x + pos_enc

    for _ in range(N_LAYERS):
        x = transformer_block(x, D_MODEL, N_HEADS, FF_DIM, dropout=DROPOUT)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def get_sinusoidal_positional_encoding(seq_len, d_model):
    # returns tensor shape (seq_len, d_model)
    pos = np.arange(seq_len)[:, None]
    i   = np.arange(d_model)[None, :]
    angle_rates = 1.0 / np.power(10000, (2*(i//2))/np.float32(d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.convert_to_tensor(pe, dtype=tf.float32)

# ======================= GroupKFold (user) =======================
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
modes_used = []
fold = 1

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)

    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    # فایل‌های train این فولد و لیبل فایل‌ها
    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}

    # انتخاب استراتیفای‌شدهٔ فایل‌های ولیدیشن
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

    # Seed-ensemble + Top-K
    val_win_probs_list, te_win_probs_list = [], []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_model((Xtr_in.shape[1], Xtr_in.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_K)

        model.fit(Xtr_in, Ytr_in,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(Xva_in, Yva_in),
                  class_weight=cw,
                  verbose=0,
                  callbacks=[es, rlr, topk])

        topk.set_topk_weights()

        val_win_probs_list.append(model.predict(Xva_in, verbose=0).ravel())
        te_win_probs_list.append(model.predict(Xte,   verbose=0).ravel())

    val_win_probs = np.mean(val_win_probs_list, axis=0)
    te_win_probs  = np.mean(te_win_probs_list,  axis=0)

    # تجمیع به سطح فایل
    val_file_probs = agg_probs_file_level(val_win_probs, Fva_in, mode=AGG_MODE, trim_q=AGG_TRIM_Q)
    te_file_probs  = agg_probs_file_level(te_win_probs,  Fte,   mode=AGG_MODE, trim_q=AGG_TRIM_Q)

    val_file_true = {int(fi): int(Y[fi]) for fi in val_file_probs.keys()}
    te_file_true  = {int(fi): int(Y[fi]) for fi in te_file_probs.keys()}

    # ===== کالیبراسیون (Platt)، اگر هر دو کلاس در val حاضر باشند
    v_files = sorted(val_file_probs.keys())
    v_p = np.array([val_file_probs[f] for f in v_files], dtype=float)[:, None]
    v_y = np.array([val_file_true[f]  for f in v_files], dtype=int)

    use_platt = (len(np.unique(v_y)) == 2)
    if use_platt:
        calib = LogisticRegression(max_iter=1000)
        calib.fit(v_p, v_y)
        # جایگزین احتمالات کالیبره در ولیدیشن و تست
        val_file_probs_cal = {f: float(calib.predict_proba([[val_file_probs[f]]])[0,1]) for f in v_files}
        t_files = sorted(te_file_probs.keys())
        te_file_probs_cal  = {f: float(calib.predict_proba([[te_file_probs[f]]])[0,1]) for f in t_files}
        val_file_probs = val_file_probs_cal
        te_file_probs  = te_file_probs_cal

    # ===== انتخاب آستانه: 1) Acc با قید Prec & PPR → 2) Balanced Acc → 3) 0.5
    picked = pick_threshold_with_constraints(val_file_probs, val_file_true,
                                             prec_floor=PREC_FLOOR,
                                             ppr_min=PPR_MIN, ppr_max=PPR_MAX)
    if picked is None:
        picked = pick_threshold_balacc(val_file_probs, val_file_true)
    thr = float(picked["thr"])
    modes_used.append(picked["mode"])
    print(f"[FOLD {fold}] Thr picked → mode={picked['mode']} thr={thr:.3f}")

    # ارزیابی فایل‌سطح روی test
    t_files_sorted = sorted(te_file_probs.keys())
    y_true = np.array([te_file_true[f]  for f in t_files_sorted], dtype=int)
    p_mean = np.array([te_file_probs[f] for f in t_files_sorted], dtype=float)
    y_pred = (p_mean > thr).astype(int)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    bal   = balanced_accuracy_score(y_true, y_pred)
    ppr_t = float(np.mean(y_pred==1))

    rows.append({"fold":fold,"thr":thr,"mode":picked["mode"],
                 "acc":acc,"precision":prec,"recall":rec,"f1":f1,"bal_acc":bal,
                 "ppr_test":ppr_t,
                 "n_test_files":len(t_files_sorted)})

    print(f"[FOLD {fold}] Test: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} BalAcc={bal:.3f} PPR={ppr_t:.2f}")
    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = {
    "WIN": WIN, "STR": STR, "Seeds": SEEDS, "TopK": TOPK_K,
    "NEG_BOOST": NEG_BOOST, "AGG_MODE": AGG_MODE, "AGG_TRIM_Q": AGG_TRIM_Q,
    "THR_RANGE": [THR_MIN, THR_MAX], "PREC_FLOOR": PREC_FLOOR,
    "PPR_RANGE_CONSTRAINT": [PPR_MIN, PPR_MAX],
    "Transformer": {"d_model": D_MODEL, "heads": N_HEADS, "layers": N_LAYERS, "ffn": FF_DIM, "dropout": DROPOUT},
    "VAL_FILE_RATIO": VAL_FILE_RATIO,
    "metrics_mean_std": {
        "Accuracy": [float(mean['acc']), float(std['acc'])],
        "Precision":[float(mean['precision']), float(std['precision'])],
        "Recall":   [float(mean['recall']), float(std['recall'])],
        "F1":       [float(mean['f1']), float(std['f1'])],
        "BalancedAcc":[float(mean['bal_acc']), float(std['bal_acc'])],
        "PPR_test": [float(mean['ppr_test']), float(std['ppr_test'])]
    },
    "per_fold_thr": [float(t) for t in df['thr'].tolist()],
    "modes_used": df['mode'].tolist()
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("== Summary ==")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("Saved to:", RESULT_DIR, flush=True)
