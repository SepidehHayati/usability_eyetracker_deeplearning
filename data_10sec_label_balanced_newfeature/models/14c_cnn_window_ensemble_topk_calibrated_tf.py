# 14c_cnn_window_ensemble_topk_calibrated_tf.py
# بهبودهای کلیدی:
# - انتخاب فایل‌های ولیدیشن به‌صورت استراتیفای‌شده (بر اساس لیبل فایل)
# - کالیبراسیون احتمال (Platt scaling) روی فایل‌های ولیدیشن + آستانه 0.5
# - گزینه‌ی جایگزین: جست‌وجوی آستانه با بهینه‌سازی F_beta (beta=0.7) بدون قید سخت Precision
# - Top-K snapshot averaging + seed-ensemble مثل قبل

import os, json, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "14c_cnn_window_ensemble_topk_calibrated_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ======================= Config =======================
WIN, STR        = 300, 150
N_SPLITS        = 6
EPOCHS          = 80
BATCH_SIZE      = 16
LR_INIT         = 1e-3
L2_REG          = 1e-5
DROPOUT_BLOCK   = 0.30
DROPOUT_HEAD    = 0.35
RANDOM_STATE    = 42
VAL_FILE_RATIO  = 0.30   # کمی بیشتر از v14b برای پایداری انتخاب آستانه/کالیبراسیون
THR_MIN, THR_MAX = 0.30, 0.70
THR_STEPS       = 41
SEEDS           = [7, 17, 23]
TOPK_K          = 5
FBETA           = 0.7    # تاکید بیشتر روی Precision نسبت به Recall

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

# ======================= Windowing =======================
def make_windows(X, Y, G, win=WIN, stride=STR):
    Xw, Yw, Gw, Fw = [], [], [], []
    for i in range(X.shape[0]):
        xi = X[i]
        for s in range(0, T - win + 1, stride):
            Xw.append(xi[s:s+win, :])
            Yw.append(Y[i]); Gw.append(G[i]); Fw.append(i)
    return np.asarray(Xw, dtype=np.float32), np.asarray(Yw, dtype=np.int64), \
           np.asarray(Gw, dtype=np.int64), np.asarray(Fw, dtype=np.int64)

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

def compute_class_weights(y):
    classes = np.unique(y)
    cnts = np.array([(y==c).sum() for c in classes], dtype=np.float32)
    n, k = len(y), len(classes)
    return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, cnts)}

def agg_probs_file_level(probs, file_ids):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    return {f: float(np.mean(ps)) for f, ps in d.items()}

def choose_files_for_validation_stratified(train_file_ids, y_file_dict, ratio, seed=0):
    """انتخاب استراتیفای‌شدهٔ فایل‌های ولیدیشن با حفظ نسبت برچسب‌ها."""
    rng = np.random.default_rng(seed)
    files0 = [f for f in train_file_ids if y_file_dict[f]==0]
    files1 = [f for f in train_file_ids if y_file_dict[f]==1]
    n_val = max(1, int(math.ceil(len(train_file_ids)*ratio)))
    n1 = max(1, int(round(n_val * (len(files1)/(len(files0)+len(files1)+1e-9)))))
    n0 = max(1, n_val - n1)
    val0 = rng.choice(files0, size=min(n0,len(files0)), replace=False).tolist() if files0 else []
    val1 = rng.choice(files1, size=min(n1,len(files1)), replace=False).tolist() if files1 else []
    val = set(val0 + val1)
    # اگر کمتر از n_val شد، از باقی مانده پر کن
    remaining = [f for f in train_file_ids if f not in val]
    while len(val) < n_val and remaining:
        val.add(remaining.pop())
    return val

def fbeta_score_binary(y_true, y_pred, beta=1.0):
    # ساده و بدون وزن: فقط برای انتخاب آستانه استفاده می‌شود
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    if tp==0 and fp==0 and fn==0:
        return 0.0
    prec = tp / (tp + fp) if tp+fp>0 else 0.0
    rec  = tp / (tp + fn) if tp+fn>0 else 0.0
    if prec==0 and rec==0:
        return 0.0
    beta2 = beta*beta
    return (1+beta2) * prec*rec / (beta2*prec + rec + 1e-12)

def choose_threshold_grid(val_file_probs, val_file_true, tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS, beta=FBETA):
    grid = np.linspace(tmin, tmax, steps)
    files = list(val_file_probs.keys())
    y_true = np.array([val_file_true[f] for f in files], dtype=int)
    p_vec  = np.array([val_file_probs[f] for f in files], dtype=float)
    best_t, best_fb = 0.5, -1.0
    for t in grid:
        y_hat = (p_vec > t).astype(int)
        fb = fbeta_score_binary(y_true, y_hat, beta=beta)
        if fb > best_fb:
            best_fb, best_t = fb, t
    return best_t, best_fb

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

# ======================= GroupKFold (user) =======================
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
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

    # class weights
    cw = compute_class_weights(Ytr_in)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # Seed-ensemble + Top-K
    val_win_probs_list, te_win_probs_list = [], []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_cnn((Xtr_in.shape[1], Xtr_in.shape[2]))
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
    val_file_probs = agg_probs_file_level(val_win_probs, Fva_in)
    te_file_probs  = agg_probs_file_level(te_win_probs,  Fte)

    # ===== روش A: کالیبراسیون (Platt) + آستانه 0.5 =====
    v_files = sorted(val_file_probs.keys())
    v_p = np.array([val_file_probs[f] for f in v_files])[:, None]  # shape (n,1)
    v_y = np.array([Y[f] for f in v_files], dtype=int)

    # اگر فقط یک کلاس در ولیدیشن بود، از روش B استفاده می‌کنیم
    use_platt = len(np.unique(v_y)) == 2
    if use_platt:
        calib = LogisticRegression(max_iter=1000)
        calib.fit(v_p, v_y)
        # روی تست
        t_files = sorted(te_file_probs.keys())
        t_p = np.array([te_file_probs[f] for f in t_files])[:, None]
        t_y = np.array([Y[f] for f in t_files], dtype=int)
        t_p_cal = calib.predict_proba(t_p)[:,1]
        y_pred = (t_p_cal > 0.5).astype(int)

        acc  = accuracy_score(t_y, y_pred)
        prec = precision_score(t_y, y_pred, zero_division=0)
        rec  = recall_score(t_y, y_pred, zero_division=0)
        f1   = f1_score(t_y, y_pred, zero_division=0)

        rows.append({"fold":fold,"mode":"calibrated","thr":0.5,
                     "acc":acc,"precision":prec,"recall":rec,"f1":f1,
                     "n_test_files":len(t_files)})
        print(f"[FOLD {fold}] Calibrated: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
    else:
        print(f"[FOLD {fold}] Calibrated mode skipped (val files single-class).")

    # ===== روش B: جست‌وجوی آستانه با F_beta روی ولیدیشن =====
    thr, fb = choose_threshold_grid(val_file_probs, {f:int(Y[f]) for f in val_file_probs.keys()},
                                    tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS, beta=FBETA)
    t_files = sorted(te_file_probs.keys())
    t_y = np.array([Y[f] for f in t_files], dtype=int)
    t_p = np.array([te_file_probs[f] for f in t_files], dtype=float)
    y_pred = (t_p > thr).astype(int)

    acc  = accuracy_score(t_y, y_pred)
    prec = precision_score(t_y, y_pred, zero_division=0)
    rec  = recall_score(t_y, y_pred, zero_division=0)
    f1   = f1_score(t_y, y_pred, zero_division=0)

    rows.append({"fold":fold,"mode":"grid_fbeta","thr":float(thr),
                 "acc":acc,"precision":prec,"recall":rec,"f1":f1,
                 "n_test_files":len(t_files)})
    print(f"[FOLD {fold}] Grid F_beta: thr={thr:.3f} (val F_beta={fb:.3f}) | "
          f"Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

def summarize_mode(df, mode):
    sub = df[df["mode"]==mode]
    if len(sub)==0: return None
    mean = sub.mean(numeric_only=True); std = sub.std(numeric_only=True)
    return {
        "mode": mode,
        "acc_mean": float(mean["acc"]), "acc_std": float(std["acc"]),
        "prec_mean": float(mean["precision"]), "prec_std": float(std["precision"]),
        "rec_mean": float(mean["recall"]), "rec_std": float(std["recall"]),
        "f1_mean": float(mean["f1"]), "f1_std": float(std["f1"]),
    }

sum_cal = summarize_mode(df, "calibrated")
sum_fbt = summarize_mode(df, "grid_fbeta")

summary = {"calibrated": sum_cal, "grid_fbeta": sum_fbt,
           "config":{"WIN":WIN,"STR":STR,"Seeds":SEEDS,"TopK":TOPK_K,
                     "VAL_FILE_RATIO":VAL_FILE_RATIO,"FBETA":FBETA}}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("[SUMMARY]", summary)
print("Saved to:", RESULT_DIR, flush=True)
