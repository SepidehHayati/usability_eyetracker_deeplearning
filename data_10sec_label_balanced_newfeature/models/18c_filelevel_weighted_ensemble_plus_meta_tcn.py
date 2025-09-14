# 18c_filelevel_weighted_ensemble_plus_meta_tcn_fixed_cm.py
# اِنسمبل وزن‌دار + استکینگ با TCN + Confusion Matrix اصلاح‌شده
# - رفع TypeError برای t_files (تبدیل به np.array)
# - اضافه کردن try-except برای جلوگیری از توقف اجرا

import os, json, math, random, itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, Input, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "18c_filelevel_weighted_ensemble_plus_meta_tcn_fixed_cm")
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
L2_REG          = 1e-4
DROPOUT_BLOCK   = 0.40
DROPOUT_HEAD    = 0.40
RANDOM_STATE    = 42
VAL_FILE_RATIO  = 0.35
AGG_MODE        = "mean"  # mean برای تست
AGG_TRIM_Q      = 0.10
THR_MIN, THR_MAX = 0.40, 0.80
THR_STEPS       = 100
SEEDS           = [7, 17, 23]
TOPK_K          = 5
NEG_BOOST       = 1.5  # 1.2 برای تعادل بهتر
USE_PLATT_CAL   = True
USE_SMOTE       = True  # غیرفعال برای تست
# تنظیمات TCN
TCN_FILTERS     = 16
TCN_KERNEL_SIZE = 3
TCN_DILATIONS   = [1, 2, 4, 8]

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

def agg_probs_window_to_file(win_probs, file_ids, mode=AGG_MODE, trim_q=AGG_TRIM_Q):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(win_probs, file_ids):
        d[int(f)].append(float(p))
    out = {}
    for f, ps in d.items():
        ps = np.array(ps, dtype=float)
        if mode=="median":
            out[f] = float(np.median(ps))
        elif mode=="trimmed":
            lo, hi = np.quantile(ps, [trim_q, 1.0-trim_q])
            clip = ps[(ps>=lo) & (ps<=hi)]
            out[f] = float(np.mean(clip)) if len(clip)>0 else float(np.mean(ps))
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

def best_threshold_for_accuracy(p, y, tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS):
    grid = np.linspace(tmin, tmax, steps)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        acc = accuracy_score(y, (p>t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc

def fraction_over_threshold(win_probs, file_ids, thr=0.5):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(win_probs, file_ids):
        d[int(f)].append(float(p))
    return {f: float((np.array(ps)>thr).mean()) for f, ps in d.items()}

def iqr_of_windows(win_probs, file_ids):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(win_probs, file_ids):
        d[int(f)].append(float(p))
    out={}
    for f, ps in d.items():
        q1,q3 = np.percentile(ps,[25,75])
        out[f] = float(q3-q1)
    return out

# ======================= Callbacks =======================
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
        Input(shape=input_shape),
        Conv1D(32, 7, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROPOUT_BLOCK),

        Conv1D(48, 5, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROPOUT_BLOCK),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_HEAD),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_tcn(input_shape, nb_filters=TCN_FILTERS, kernel_size=TCN_KERNEL_SIZE, dilations=TCN_DILATIONS):
    inp = Input(shape=input_shape)
    x = inp
    for d in dilations:
        x1 = Conv1D(nb_filters, kernel_size, padding='causal', dilation_rate=d,
                    kernel_regularizer=regularizers.l2(L2_REG))(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(DROPOUT_BLOCK)(x1)
        x_adjusted = Conv1D(nb_filters, 1, padding='same',
                           kernel_regularizer=regularizers.l2(L2_REG))(x)
        x = Add()([x_adjusted, x1])
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = Dropout(DROPOUT_HEAD)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_inceptiontime(input_shape):
    inp = Input(shape=input_shape)
    x = inception_module(inp, 16); x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x); x = Dropout(DROPOUT_BLOCK)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x); x = Dropout(DROPOUT_HEAD)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def inception_module(x, nb_filters):
    t1 = Conv1D(nb_filters, 1, padding='same', activation='relu')(x)
    t2 = Conv1D(nb_filters, 3, padding='same', activation='relu')(x)
    t3 = Conv1D(nb_filters, 5, padding='same', activation='relu')(x)
    t4 = MaxPooling1D(3, strides=1, padding='same')(x)
    t4 = Conv1D(nb_filters, 1, padding='same', activation='relu')(t4)
    return tf.keras.layers.Concatenate()([t1,t2,t3,t4])

def fit_seed_ensemble(model_fn, Xtr_in, Ytr_in, Xva_in, Xte, class_weight, seeds=SEEDS):
    val_win_probs_list, te_win_probs_list = [], []
    for sd in seeds:
        tf.keras.backend.clear_session()
        np.random.seed(sd); tf.random.set_seed(sd)
        model = model_fn((Xtr_in.shape[1], Xtr_in.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_K)
        model.fit(Xtr_in, Ytr_in,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(Xva_in, Yva_in),
                  class_weight=class_weight,
                  verbose=0,
                  callbacks=[es, rlr, topk])
        topk.set_topk_weights()
        val_win_probs_list.append(model.predict(Xva_in, verbose=0).ravel())
        te_win_probs_list.append(model.predict(Xte,   verbose=0).ravel())
    return np.mean(val_win_probs_list, axis=0), np.mean(te_win_probs_list, axis=0)

# ======================= Main CV Loop =======================
gkf = GroupKFold(n_splits=N_SPLITS)
rows = []
fold = 1

weight_candidates = [
    (0.2,0.5,0.3), (0.25,0.5,0.25), (0.3,0.4,0.3), (0.3,0.5,0.2),
    (0.4,0.4,0.2), (0.4,0.3,0.3), (0.2,0.4,0.4), (0.33,0.34,0.33),
]

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

    # اعمال SMOTE (اگر فعال باشد)
    if USE_SMOTE:
        smote = SMOTE(random_state=RANDOM_STATE + fold)
        Xtr_in_reshaped = Xtr_in.reshape(-1, WIN * C)
        Xtr_in_res, Ytr_in_res = smote.fit_resample(Xtr_in_reshaped, Ytr_in)
        Xtr_in = Xtr_in_res.reshape(-1, WIN, C)
        Ytr_in = Ytr_in_res

    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # سه مدل پایه: CNN / TCN / InceptionTime
    val_cnn, te_cnn = fit_seed_ensemble(build_cnn,          Xtr_in, Ytr_in, Xva_in, Xte, cw, SEEDS)
    val_tcn, te_tcn = fit_seed_ensemble(build_tcn,          Xtr_in, Ytr_in, Xva_in, Xte, cw, SEEDS)
    val_inc, te_inc = fit_seed_ensemble(build_inceptiontime, Xtr_in, Ytr_in, Xva_in, Xte, cw, SEEDS)

    # فایل‌سطح
    val_file_cnn = agg_probs_window_to_file(val_cnn, Fva_in, AGG_MODE, AGG_TRIM_Q)
    val_file_tcn = agg_probs_window_to_file(val_tcn, Fva_in, AGG_MODE, AGG_TRIM_Q)
    val_file_inc = agg_probs_window_to_file(val_inc, Fva_in, AGG_MODE, AGG_TRIM_Q)

    te_file_cnn = agg_probs_window_to_file(te_cnn, Fte, AGG_MODE, AGG_TRIM_Q)
    te_file_tcn = agg_probs_window_to_file(te_tcn, Fte, AGG_MODE, AGG_TRIM_Q)
    te_file_inc = agg_probs_window_to_file(te_inc, Fte, AGG_MODE, AGG_TRIM_Q)

    # کالیبراسیون Platt
    def platt_fit_apply(val_dict, te_dict):
        files_v = sorted(val_dict.keys())
        pv = np.array([val_dict[f] for f in files_v])[:,None]
        yv = np.array([Y[f] for f in files_v], dtype=int)
        ok = len(np.unique(yv))==2 and USE_PLATT_CAL
        if not ok:
            return val_dict, te_dict
        cal = LogisticRegression(max_iter=1000)
        cal.fit(pv, yv)
        files_t = sorted(te_dict.keys())
        pt = np.array([te_dict[f] for f in files_t])[:,None]
        val_cal = {f: float(cal.predict_proba(np.array([[val_dict[f]]]))[0,1]) for f in files_v}
        te_cal  = {f: float(p) for f, p in zip(files_t, cal.predict_proba(pt)[:,1])}
        return val_cal, te_cal

    val_file_cnn, te_file_cnn = platt_fit_apply(val_file_cnn, te_file_cnn)
    val_file_tcn, te_file_tcn = platt_fit_apply(val_file_tcn, te_file_tcn)
    val_file_inc, te_file_inc = platt_fit_apply(val_file_inc, te_file_inc)

    # آرایه‌های کارا
    v_files = sorted(set(val_file_cnn.keys()) & set(val_file_tcn.keys()) & set(val_file_inc.keys()))
    t_files = sorted(set(te_file_cnn.keys())  & set(te_file_tcn.keys())  & set(te_file_inc.keys()))
    v_y = np.array([Y[f] for f in v_files], dtype=int)
    t_y = np.array([Y[f] for f in t_files], dtype=int)

    v_cnn = np.array([val_file_cnn[f] for f in v_files])
    v_tcn = np.array([val_file_tcn[f] for f in v_files])
    v_inc = np.array([val_file_inc[f] for f in v_files])

    t_cnn = np.array([te_file_cnn[f] for f in t_files])
    t_tcn = np.array([te_file_tcn[f] for f in t_files])
    t_inc = np.array([te_file_inc[f] for f in t_files])

    # (A) اِنسمبل وزن‌دار
    best_val_acc_w, best_w, best_thr_w = -1.0, (1/3,1/3,1/3), 0.5
    for (wc, wt, wi) in weight_candidates:
        v_ens = wc*v_cnn + wt*v_tcn + wi*v_inc
        thr, acc_val = best_threshold_for_accuracy(v_ens, v_y, THR_MIN, THR_MAX, THR_STEPS)
        if acc_val > best_val_acc_w:
            best_val_acc_w, best_w, best_thr_w = acc_val, (wc,wt,wi), thr
    t_ens = best_w[0]*t_cnn + best_w[1]*t_tcn + best_w[2]*t_inc
    y_pred_w = (t_ens > best_thr_w).astype(int)
    acc_w  = accuracy_score(t_y, y_pred_w)
    prec_w = precision_score(t_y, y_pred_w, zero_division=0)
    rec_w  = recall_score(t_y, y_pred_w, zero_division=0)
    f1_w   = f1_score(t_y, y_pred_w, zero_division=0)

    print(f"[FOLD {fold}] Weighted Ensembling → wCNN={best_w[0]:.2f}, wTCN={best_w[1]:.2f}, wINC={best_w[2]:.2f} | thr={best_thr_w:.3f} | Acc(val)={best_val_acc_w:.3f}")
    print(f"[FOLD {fold}] Weighted Test: Acc={acc_w:.3f} Prec={prec_w:.3f} Rec={rec_w:.3f} F1={f1_w:.3f}")

    # (B) استکینگ با XGBoost
    def build_meta_features(vc, vt, vi):
        Xmeta = np.stack([vc, vt, vi], axis=1)
        mean = Xmeta.mean(1); mx = Xmeta.max(1); mn = Xmeta.min(1); var = Xmeta.var(1)
        return np.stack([vc, vt, vi, mean, mx, mn, var], axis=1)

    def build_window_meta(F_ids, p_win_c, p_win_t, p_win_i, files):
        frac_c = fraction_over_threshold(p_win_c, F_ids, 0.5)
        frac_t = fraction_over_threshold(p_win_t, F_ids, 0.5)
        frac_i = fraction_over_threshold(p_win_i, F_ids, 0.5)
        iqr_c  = iqr_of_windows(p_win_c, F_ids)
        iqr_t  = iqr_of_windows(p_win_t, F_ids)
        iqr_i  = iqr_of_windows(p_win_i, F_ids)
        feat = []
        for f in files:
            feat.append([frac_c[f], frac_t[f], frac_i[f], iqr_c[f], iqr_t[f], iqr_i[f]])
        return np.array(feat, dtype=float)

    Xmeta_v_basic = build_meta_features(v_cnn, v_tcn, v_inc)
    Xmeta_t_basic = build_meta_features(t_cnn, t_tcn, t_inc)

    Xmeta_v_win = build_window_meta(Fva_in, val_cnn, val_tcn, val_inc, v_files)
    Xmeta_t_win = build_window_meta(Fte,    te_cnn, te_tcn, te_inc, t_files)

    Xmeta_v = np.concatenate([Xmeta_v_basic, Xmeta_v_win], axis=1)
    Xmeta_t = np.concatenate([Xmeta_t_basic, Xmeta_t_win], axis=1)

    stack_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)
    stack_xgb.fit(Xmeta_v, v_y)
    p_stack = stack_xgb.predict_proba(Xmeta_t)[:,1]
    thr_s, _ = best_threshold_for_accuracy(p_stack, t_y, THR_MIN, THR_MAX, THR_STEPS)
    y_pred_s = (p_stack > thr_s).astype(int)
    acc_s  = accuracy_score(t_y, y_pred_s)
    prec_s = precision_score(t_y, y_pred_s, zero_division=0)
    rec_s  = recall_score(t_y, y_pred_s, zero_division=0)
    f1_s   = f1_score(t_y, y_pred_s, zero_division=0)

    print(f"[FOLD {fold}] Stacking XGBoost → thr={thr_s:.3f} | Test: Acc={acc_s:.3f} Prec={prec_s:.3f} Rec={rec_s:.3f} F1={f1_s:.3f}")

    # اضافه کردن Confusion Matrix و لیست خطاها (با try-except برای جلوگیری از توقف)
    try:
        t_files_np = np.array(t_files)  # تبدیل به np.array
        cm_w = confusion_matrix(t_y, y_pred_w)
        cm_s = confusion_matrix(t_y, y_pred_s)
        print(f"[FOLD {fold}] Weighted Confusion Matrix:\n{cm_w}")
        print(f"[FOLD {fold}] Stacking Confusion Matrix:\n{cm_s}")
        errors_w = t_files_np[t_y != y_pred_w]
        errors_s = t_files_np[t_y != y_pred_s]
        print(f"[FOLD {fold}] Weighted Errors: {errors_w.tolist()}")
        print(f"[FOLD {fold}] Stacking Errors: {errors_s.tolist()}")
    except Exception as e:
        print(f"[FOLD {fold}] Error in Confusion Matrix/Errors: {e}")

    # انتخاب روش برنده
    v_stack = stack_xgb.predict_proba(Xmeta_v)[:,1]
    thr_sv, acc_sv = best_threshold_for_accuracy(v_stack, v_y, THR_MIN, THR_MAX, THR_STEPS)

    picked_mode = "weighted" if best_val_acc_w >= acc_sv else "stack_xgb"
    if picked_mode=="weighted":
        rows.append({"fold":fold,"mode":"weighted","acc":acc_w,"precision":prec_w,"recall":rec_w,"f1":f1_w,
                     "thr":best_thr_w,"w_cnn":best_w[0],"w_tcn":best_w[1],"w_inc":best_w[2]})
        print(f"[FOLD {fold}] >> PICKED: Weighted (val Acc={best_val_acc_w:.3f} >= Stack val Acc={acc_sv:.3f})")
    else:
        rows.append({"fold":fold,"mode":"stack_xgb","acc":acc_s,"precision":prec_s,"recall":rec_s,"f1":f1_s,
                     "thr":thr_s,"w_cnn":np.nan,"w_tcn":np.nan,"w_inc":np.nan})
        print(f"[FOLD {fold}] >> PICKED: Stack XGBoost (val Acc={acc_sv:.3f} > Weighted val Acc={best_val_acc_w:.3f})")

    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

def summarize(df, mode):
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

sum_weighted = summarize(df, "weighted")
sum_stack    = summarize(df, "stack_xgb")
sum_all = {
    "weighted": sum_weighted,
    "stack_xgb": sum_stack,
    "config": {
        "WIN":WIN,"STR":STR,"Seeds":SEEDS,"TopK":TOPK_K,"NEG_BOOST":NEG_BOOST,
        "AGG_MODE":AGG_MODE,"AGG_TRIM_Q":AGG_TRIM_Q,"VAL_FILE_RATIO":VAL_FILE_RATIO,
        "THR_RANGE":[THR_MIN,THR_MAX,THR_STEPS],
        "USE_PLATT_CAL": USE_PLATT_CAL,
        "USE_SMOTE": USE_SMOTE,
        "TCN":{"filters":TCN_FILTERS,"kernel_size":TCN_KERNEL_SIZE,"dilations":TCN_DILATIONS},
        "weight_candidates": weight_candidates,
    }
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(sum_all, f, indent=2, ensure_ascii=False)

print("\n[SUMMARY]")
print(json.dumps(sum_all, indent=2, ensure_ascii=False))
print("Saved to:", RESULT_DIR, flush=True)