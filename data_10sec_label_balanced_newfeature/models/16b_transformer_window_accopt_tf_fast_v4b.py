# 16b_transformer_window_accopt_tf_fast_v4b.py
# اصلاحات نسبت به v4:
# - THR_MIN..THR_MAX = 0.40..0.80
# - MIN_C0_RATIO = None (بدون قید سهم کلاس 0)
# - NEG_BOOST = 1.4
# - VAL_FILE_RATIO = 0.35
# - انتخاب آستانه: بیشینه‌سازی Accuracy با قید Recall >= 0.75؛
#   اگر قید پاس نشد، آستانه‌ی بهترین F1.
# - باقی معماری همان نسخه‌ی fast (Transformer یک‌لایه)

import os, math, random, json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                     MultiHeadAttention, Conv1D, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "16b_transformer_window_accopt_tf_fast_v4b")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ======================= Config =======================
WIN, STR        = 500, 200
N_SPLITS        = 6
EPOCHS          = 60
BATCH_SIZE      = 16
LR_INIT         = 1e-3
L2_REG          = 1e-5
DROPOUT         = 0.20
RANDOM_STATE    = 42
VAL_FILE_RATIO  = 0.35
THR_MIN, THR_MAX = 0.40, 0.80
THR_STEPS       = 41
MIN_C0_RATIO    = None      # بدون قید سهم کلاس 0
RECALL_FLOOR    = 0.75      # قید برای انتخاب آستانه
SEEDS           = [7, 17, 23]
TOPK_K          = 5
NEG_BOOST       = 1.4
AGG_MODE        = "trimmed" # "trimmed" یا "median"
TRIM_FRAC       = 0.10

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

def aggregate_probs_window_to_file(win_probs, file_ids, mode=AGG_MODE, trim_frac=TRIM_FRAC):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(win_probs, file_ids):
        d[int(f)].append(float(p))
    out = {}
    for f, lst in d.items():
        arr = np.array(lst, dtype=float)
        if mode == "median":
            out[f] = float(np.median(arr))
        elif mode == "trimmed":
            n = len(arr)
            if n >= 4:
                k = int(np.floor(trim_frac * n))
                arr = np.sort(arr)
                trimmed = arr[k:n-k] if (n - 2*k) > 0 else arr
                out[f] = float(np.mean(trimmed))
            else:
                out[f] = float(np.mean(arr))
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

def choose_threshold_acc_with_recall(val_probs_dict, val_true_dict,
                                     tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS,
                                     recall_floor=RECALL_FLOOR):
    grid = np.linspace(tmin, tmax, steps)
    files = list(val_probs_dict.keys())
    y_true = np.array([val_true_dict[f] for f in files], dtype=int)
    p_vec  = np.array([val_probs_dict[f] for f in files], dtype=float)

    # کاندیدهایی که Recall >= floor دارند
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        y_hat = (p_vec > t).astype(int)
        rec = recall_score(y_true, y_hat, zero_division=0)
        if rec >= recall_floor:
            acc = accuracy_score(y_true, y_hat)
            if acc > best_acc:
                best_acc, best_t = acc, t

    if best_acc >= 0:
        return best_t, best_acc, True

    # اگر هیچ آستانه‌ای قید Recall را پاس نکرد، بهترین F1 را بردار
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_hat = (p_vec > t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    acc = accuracy_score(y_true, (p_vec > best_t).astype(int))
    return best_t, acc, False

# ======================= Positional Encoding =======================
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        d_model = tf.cast(self.d_model, tf.float32)
        pos = tf.cast(tf.range(T), tf.float32)[:, tf.newaxis]
        i   = tf.cast(tf.range(self.d_model), tf.float32)[tf.newaxis, :]
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i/2.0)) / d_model)
        angles = pos * angle_rates
        sines = tf.sin(angles[:, 0::2])
        cosines = tf.cos(angles[:, 1::2])
        def interleave(a, b):
            a_exp = tf.expand_dims(a, axis=-1)
            b_exp = tf.expand_dims(b, axis=-1)
            stacked = tf.concat([a_exp, b_exp], axis=-1)
            return tf.reshape(stacked, (tf.shape(a)[0], tf.shape(a)[1]*2))
        pe = interleave(sines, cosines)
        pe = tf.cast(pe, x.dtype)
        pe = tf.expand_dims(pe, axis=0)
        return x + pe

# ======================= Transformer (fast) =======================
def build_transformer(input_shape, d_model=64, num_heads=4, ffn_dim=128, num_layers=1,
                      dropout=DROPOUT, l2=L2_REG, lr=LR_INIT):
    inp = Input(shape=input_shape)        # (T, C)
    x = Conv1D(d_model, kernel_size=1, padding='same',
               kernel_regularizer=regularizers.l2(l2))(inp)
    x = PositionalEncoding(d_model)(x)
    for _ in range(num_layers):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads,
                                      dropout=dropout)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_out)
        ffn = tf.keras.Sequential([
            Dense(ffn_dim, activation='relu'),
            Dropout(dropout),
            Dense(d_model)
        ])
        ffn_out = ffn(x)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_out)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

# ======================= GroupKFold (user) =======================
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
fold = 1

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)

    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}

    # Stratified file-level validation split
    val_files = choose_files_for_validation_stratified(train_files, y_file, VAL_FILE_RATIO,
                                                       seed=RANDOM_STATE + fold)
    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # standardize
    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    # class weights (+NEG_BOOST)
    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # Seed-ensemble + Top-K
    val_win_probs_list, te_win_probs_list = [], []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_transformer((Xtr_in.shape[1], Xtr_in.shape[2]),
                                  d_model=64, num_heads=4, ffn_dim=128, num_layers=1,
                                  dropout=DROPOUT, l2=L2_REG, lr=LR_INIT)
        es  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
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

    # file-level aggregation
    val_file_probs = aggregate_probs_window_to_file(val_win_probs, Fva_in, mode=AGG_MODE)
    te_file_probs  = aggregate_probs_window_to_file(te_win_probs,  Fte,   mode=AGG_MODE)

    # Platt calibration
    v_files = sorted(val_file_probs.keys())
    v_p = np.array([val_file_probs[f] for f in v_files])[:, None]
    v_y = np.array([Y[int(f)] for f in v_files], dtype=int)
    use_platt = len(np.unique(v_y)) == 2
    if use_platt:
        calib = LogisticRegression(max_iter=1000)
        calib.fit(v_p, v_y)
        val_file_probs_cal = {f: calib.predict_proba([[val_file_probs[f]]])[0,1] for f in v_files}
        t_files = sorted(te_file_probs.keys())
        te_file_probs_cal = {f: calib.predict_proba([[te_file_probs[f]]])[0,1] for f in t_files}
    else:
        val_file_probs_cal = val_file_probs
        te_file_probs_cal  = te_file_probs

    # Threshold selection: max Acc with Recall>=RECALL_FLOOR; else best F1
    val_file_true = {int(f): int(Y[int(f)]) for f in val_file_probs_cal.keys()}
    thr, acc_val, ok = choose_threshold_acc_with_recall(val_file_probs_cal, val_file_true,
                                                        tmin=THR_MIN, tmax=THR_MAX,
                                                        steps=THR_STEPS, recall_floor=RECALL_FLOOR)
    print(f"[FOLD {fold}] ThrOpt: thr={thr:.3f} (val Acc={acc_val:.3f}, recall_floor_ok={ok})")

    # test
    t_files = sorted(te_file_probs_cal.keys())
    t_y = np.array([Y[int(f)] for f in t_files], dtype=int)
    t_p = np.array([te_file_probs_cal[f] for f in t_files], dtype=float)
    y_pred = (t_p > thr).astype(int)

    acc  = accuracy_score(t_y, y_pred)
    prec = precision_score(t_y, y_pred, zero_division=0)
    rec  = recall_score(t_y, y_pred, zero_division=0)
    f1   = f1_score(t_y, y_pred, zero_division=0)

    rows.append({"fold":fold, "thr":float(thr),
                 "acc":acc, "precision":prec, "recall":rec, "f1":f1,
                 "n_test_files":len(t_files)})

    print(f"[FOLD {fold}] Test: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = {
    "Accuracy": [float(mean["acc"]), float(std["acc"])],
    "Precision":[float(mean["precision"]), float(std["precision"])],
    "Recall":   [float(mean["recall"]), float(std["recall"])],
    "F1":       [float(mean["f1"]), float(std["f1"])],
    "Config": {
        "WIN": WIN, "STR": STR, "Seeds": SEEDS, "TopK": TOPK_K, "NEG_BOOST": NEG_BOOST,
        "AGG_MODE": AGG_MODE, "TRIM_FRAC": TRIM_FRAC,
        "VAL_FILE_RATIO": VAL_FILE_RATIO,
        "THR_MIN": THR_MIN, "THR_MAX": THR_MAX, "THR_STEPS": THR_STEPS,
        "RECALL_FLOOR": RECALL_FLOOR
    }
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("== Calibrated + Acc-Opt with Recall floor ==")
print(f"Accuracy: {summary['Accuracy'][0]:.4f} ± {summary['Accuracy'][1]:.4f}")
print(f"Precision: {summary['Precision'][0]:.4f} ± {summary['Precision'][1]:.4f}")
print(f"Recall: {summary['Recall'][0]:.4f} ± {summary['Recall'][1]:.4f}")
print(f"F1: {summary['F1'][0]:.4f} ± {summary['F1'][1]:.4f}")
print("Saved to:", RESULT_DIR, flush=True)
