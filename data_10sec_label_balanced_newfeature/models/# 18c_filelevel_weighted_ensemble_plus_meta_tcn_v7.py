# 18c_filelevel_weighted_ensemble_plus_meta_tcn_v7.py
# تغییرات: NEG_BOOST=1.5, USE_SMOTE=False, گسترش param_grid, افزایش وزن TCN

import os, json, math, random, itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
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
RESULT_DIR = os.path.join("..", "results", "18c_filelevel_weighted_ensemble_plus_meta_tcn_v7")
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
AGG_MODE        = "median"
AGG_TRIM_Q      = 0.10
THR_MIN, THR_MAX = 0.40, 0.80
THR_STEPS       = 100
SEEDS           = [7, 17, 23, 29, 37]
TOPK_K          = 5
NEG_BOOST       = 1.5  # بازگشت به بهترین مقدار
USE_PLATT_CAL   = True
USE_SMOTE       = False  # غیرفعال کردن SMOTE
TCN_FILTERS     = 16
TCN_KERNEL_SIZE = 3
TCN_DILATIONS   = [1, 2, 4, 8]

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

# ======================= Functions =======================
def make_windows(X, Y, G, win, stride):
    N, T, C = X.shape
    Xw, Yw, Gw, Fw = [], [], [], []
    for i in range(N):
        t = 0
        while t + win <= T:
            Xw.append(X[i, t:t+win])
            Yw.append(Y[i])
            Gw.append(G[i])
            Fw.append(i)
            t += stride
    Xw = np.array(Xw)
    Yw = np.array(Yw)
    Gw = np.array(Gw)
    Fw = np.array(Fw)
    print(f"[WINDOWS] Xw={Xw.shape}, Yw={Yw.shape}, Gw={Gw.shape}, Fw={Fw.shape}", flush=True)
    return Xw, Yw, Gw, Fw

def standardize_per_fold(Xtr, Xva, Xte):
    sc = StandardScaler()
    Xtr_shape = Xtr.shape
    Xva_shape = Xva.shape
    Xte_shape = Xte.shape
    Xtr = sc.fit_transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr_shape)
    Xva = sc.transform(Xva.reshape(-1, Xva.shape[-1])).reshape(Xva_shape)
    Xte = sc.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte_shape)
    return Xtr, Xva, Xte

def compute_class_weights(Y, neg_boost=1.0):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(Y)
    weights = compute_class_weight('balanced', classes=classes, y=Y)
    cw = {0: weights[0] * neg_boost, 1: weights[1]}
    return cw

def agg_probs_window_to_file(probs, F_ids, mode="median", trim_q=0.0):
    files = np.unique(F_ids)
    file_probs = {}
    for f in files:
        mask = F_ids == f
        p = probs[mask]
        if trim_q > 0:
            q_low, q_high = np.quantile(p, [trim_q, 1-trim_q])
            p = p[(p >= q_low) & (p <= q_high)]
        if mode == "median":
            file_probs[f] = np.median(p)
        elif mode == "mean":
            file_probs[f] = np.mean(p)
    return file_probs

def choose_files_for_validation_stratified(files, y_file, val_ratio, seed):
    from sklearn.model_selection import train_test_split
    files_0 = [f for f in files if y_file[f] == 0]
    files_1 = [f for f in files if y_file[f] == 1]
    val_files_0, _ = train_test_split(files_0, test_size=(1-val_ratio), random_state=seed)
    val_files_1, _ = train_test_split(files_1, test_size=(1-val_ratio), random_state=seed)
    return sorted(val_files_0 + val_files_1)

def best_threshold_for_accuracy(probs, y_true, thr_min, thr_max, n_steps):
    thresholds = np.linspace(thr_min, thr_max, n_steps)
    best_acc, best_thr = -1, 0
    for thr in thresholds:
        y_pred = (probs > thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc

def fraction_over_threshold(probs, F_ids, thr):
    files = np.unique(F_ids)
    frac = {}
    for f in files:
        mask = F_ids == f
        frac[f] = np.mean(probs[mask] > thr)
    return frac

def iqr_of_windows(probs, F_ids):
    files = np.unique(F_ids)
    iqr = {}
    for f in files:
        mask = F_ids == f
        p = probs[mask]
        iqr[f] = np.percentile(p, 75) - np.percentile(p, 25)
    return iqr

class TopKSaver(tf.keras.callbacks.Callback):
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.top_k_weights = []
        self.top_k_losses = []

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if len(self.top_k_losses) < self.k or val_loss < max(self.top_k_losses):
            if len(self.top_k_losses) >= self.k:
                idx = np.argmax(self.top_k_losses)
                self.top_k_losses.pop(idx)
                self.top_k_weights.pop(idx)
            self.top_k_losses.append(val_loss)
            weights = self.model.get_weights()
            self.top_k_weights.append(weights)
        if len(self.top_k_losses) > 0 and val_loss <= min(self.top_k_losses):
            self.model.set_weights(self.top_k_weights[np.argmin(self.top_k_losses)])

def build_cnn():
    model = Sequential([
        Conv1D(16, 5, padding='same', input_shape=(WIN, C), kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2),
        Conv1D(32, 5, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2),
        Dropout(DROPOUT_BLOCK),
        Conv1D(64, 5, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
        Dropout(DROPOUT_HEAD),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_tcn():
    inputs = Input(shape=(WIN, C))
    x = inputs
    for i, dilation in enumerate(TCN_DILATIONS):
        x = Conv1D(TCN_FILTERS, TCN_KERNEL_SIZE, padding='causal', dilation_rate=dilation,
                   kernel_regularizer=regularizers.l2(L2_REG))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if i < len(TCN_DILATIONS) - 1:
            x = Dropout(DROPOUT_BLOCK)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = Dropout(DROPOUT_HEAD)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_inceptiontime():
    inputs = Input(shape=(WIN, C))
    x = inputs
    for _ in range(2):
        x1 = Conv1D(16, 1, padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
        x2 = Conv1D(16, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
        x3 = Conv1D(16, 5, padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
        x4 = MaxPooling1D(3, strides=1, padding='same')(x)
        x4 = Conv1D(16, 1, padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x4)
        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(DROPOUT_BLOCK)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = Dropout(DROPOUT_HEAD)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def fit_seed_ensemble(build_fn, Xtr, Ytr, Xva, Yva, cw, seeds):
    val_probs, te_probs = [], []
    for seed in seeds:
        tf.random.set_seed(seed)
        model = build_fn()
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            TopKSaver(k=TOPK_K)
        ]
        print(f"[DEBUG] Xtr shape: {Xtr.shape}, Ytr shape: {Ytr.shape}, Xva shape: {Xva.shape}, Yva shape: {Yva.shape}")
        model.fit(Xtr, Ytr, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(Xva, Yva),
                  callbacks=callbacks, class_weight=cw, verbose=0)
        val_probs.append(model.predict(Xva, batch_size=BATCH_SIZE, verbose=0)[:,0])
        te_probs.append(model.predict(Xte, batch_size=BATCH_SIZE, verbose=0)[:,0])
    return np.mean(val_probs, axis=0), np.mean(te_probs, axis=0)

# ======================= Windowing =======================
Xw, Yw, Gw, Fw = make_windows(X, Y, G, WIN, STR)

# ======================= Main CV Loop =======================
gkf = GroupKFold(n_splits=N_SPLITS)
rows = []
fold = 1

weight_candidates = [
    (0.2, 0.6, 0.2), (0.25, 0.55, 0.2), (0.3, 0.5, 0.2), (0.2, 0.5, 0.3),
    (0.3, 0.4, 0.3), (0.4, 0.4, 0.2), (0.2, 0.4, 0.4), (0.33, 0.34, 0.33)
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
    va_mask = np.array([f in val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    print(f"[DEBUG] Before SMOTE: Xtr_in shape: {Xtr_in.shape}, Ytr_in shape: {Ytr_in.shape}")
    if USE_SMOTE:
        smote = SMOTE(random_state=RANDOM_STATE + fold)
        Xtr_in_reshaped = Xtr_in.reshape(-1, WIN * C)
        Xtr_in_res, Ytr_in_res = smote.fit_resample(Xtr_in_reshaped, Ytr_in)
        Xtr_in = Xtr_in_res.reshape(-1, WIN, C)
        Ytr_in = Ytr_in_res
        print(f"[DEBUG] After SMOTE: Xtr_in shape: {Xtr_in.shape}, Ytr_in shape: {Ytr_in.shape}")

    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    val_cnn, te_cnn = fit_seed_ensemble(build_cnn, Xtr_in, Ytr_in, Xva_in, Yva_in, cw, SEEDS)
    val_tcn, te_tcn = fit_seed_ensemble(build_tcn, Xtr_in, Ytr_in, Xva_in, Yva_in, cw, SEEDS)
    val_inc, te_inc = fit_seed_ensemble(build_inceptiontime, Xtr_in, Ytr_in, Xva_in, Yva_in, cw, SEEDS)

    val_file_cnn = agg_probs_window_to_file(val_cnn, Fva_in, AGG_MODE, AGG_TRIM_Q)
    val_file_tcn = agg_probs_window_to_file(val_tcn, Fva_in, AGG_MODE, AGG_TRIM_Q)
    val_file_inc = agg_probs_window_to_file(val_inc, Fva_in, AGG_MODE, AGG_TRIM_Q)

    te_file_cnn = agg_probs_window_to_file(te_cnn, Fte, AGG_MODE, AGG_TRIM_Q)
    te_file_tcn = agg_probs_window_to_file(te_tcn, Fte, AGG_MODE, AGG_TRIM_Q)
    te_file_inc = agg_probs_window_to_file(te_inc, Fte, AGG_MODE, AGG_TRIM_Q)

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

    v_files = sorted(set(val_file_cnn.keys()) & set(val_file_tcn.keys()) & set(val_file_inc.keys()))
    t_files = sorted(set(te_file_cnn.keys()) & set(te_file_tcn.keys()) & set(te_file_inc.keys()))
    v_y = np.array([Y[f] for f in v_files], dtype=int)
    t_y = np.array([Y[f] for f in t_files], dtype=int)

    v_cnn = np.array([val_file_cnn[f] for f in v_files])
    v_tcn = np.array([val_file_tcn[f] for f in v_files])
    v_inc = np.array([val_file_inc[f] for f in v_files])

    t_cnn = np.array([te_file_cnn[f] for f in t_files])
    t_tcn = np.array([te_file_tcn[f] for f in t_files])
    t_inc = np.array([te_file_inc[f] for f in t_files])

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
    Xmeta_t_win = build_window_meta(Fte, te_cnn, te_tcn, te_inc, t_files)

    Xmeta_v = np.concatenate([Xmeta_v_basic, Xmeta_v_win], axis=1)
    Xmeta_t = np.concatenate([Xmeta_t_basic, Xmeta_t_win], axis=1)

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    stack_xgb = GridSearchCV(XGBClassifier(random_state=RANDOM_STATE), param_grid, cv=3)
    stack_xgb.fit(Xmeta_v, v_y)
    print(f"[FOLD {fold}] Best XGBoost params: {stack_xgb.best_params_}")
    p_stack = stack_xgb.best_estimator_.predict_proba(Xmeta_t)[:,1]
    thr_s, _ = best_threshold_for_accuracy(p_stack, t_y, THR_MIN, THR_MAX, THR_STEPS)
    y_pred_s = (p_stack > thr_s).astype(int)
    acc_s  = accuracy_score(t_y, y_pred_s)
    prec_s = precision_score(t_y, y_pred_s, zero_division=0)
    rec_s  = recall_score(t_y, y_pred_s, zero_division=0)
    f1_s   = f1_score(t_y, y_pred_s, zero_division=0)

    print(f"[FOLD {fold}] Stacking XGBoost → thr={thr_s:.3f} | Test: Acc={acc_s:.3f} Prec={prec_s:.3f} Rec={rec_s:.3f} F1={f1_s:.3f}")

    try:
        t_files_np = np.array(t_files)
        print(f"[FOLD {fold}] Test Files: {t_files}")
        cm_w = confusion_matrix(t_y, y_pred_w)
        cm_s = confusion_matrix(t_y, y_pred_s)
        print(f"[FOLD {fold}] Weighted Confusion Matrix:\n{cm_w}")
        print(f"[FOLD {fold}] Stacking Confusion Matrix:\n{cm_s}")
        errors_w = t_files_np[t_y != y_pred_w]
        errors_s = t_files_np[t_y != y_pred_s]
        print(f"[FOLD {fold}] Weighted Errors: {errors_w.tolist()}")
        print(f"[FOLD {fold}] Stacking Errors: {errors_s.tolist()}")
        error_data = pd.DataFrame({
            'fold': [fold],
            'test_files': [t_files],
            'weighted_errors': [errors_w.tolist()],
            'stacking_errors': [errors_s.tolist()]
        })
        error_data.to_csv(os.path.join(RESULT_DIR, f"fold_{fold}_errors.csv"), index=False)
    except Exception as e:
        print(f"[FOLD {fold}] Error in Confusion Matrix/Errors: {e}")

    v_stack = stack_xgb.best_estimator_.predict_proba(Xmeta_v)[:,1]
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
sum_stack = summarize(df, "stack_xgb")
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