# 18i_filelevel_ensemble_robust_adaptthr_tf.py
# Robust file-level ensemble for Gaze task:
# - Per-user standardization
# - Aggregation sweep (median/trimmed + trim_q)
# - Calibration picked via CV (Platt vs Isotonic) per base model
# - Three ensemble heads:
#     * M1: LR stacking with meta features (window stats too)
#     * M2: Weighted average (grid search)
#     * M3: Rank-average (robust to calibration shift)
# - Thresholding:
#     * Validation: Balanced Accuracy with F1-macro tie-break (OOF where needed)
#     * Test: Label-free POS-RATE ADAPTATION (quantile) + anti-single-class guard
# - No leakage of test labels. Safer class weighting.

import os, json, math, random, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, Input, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "18i_filelevel_ensemble_robust_adaptthr_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)  # user/group per file
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
AGG_MODE_CAND   = ["median", "trimmed"]
TRIM_Q_CAND     = [0.00, 0.05, 0.10, 0.15, 0.20]
THR_MIN, THR_MAX = 0.40, 0.80
THR_STEPS       = 41
SEEDS           = [7, 17, 23]
TOPK_K          = 5
NEG_BOOST       = 1.1
CAL_CV_SPLITS   = 3
LR_CV_SPLITS    = 3
MIN_ISO_SAMPLES = 15
# Test-time POS rate adaptation (clip to avoid extremes)
ADAPT_POS_CLIP  = (0.20, 0.80)
RANK_EPS        = 1e-9

TRF_DMODEL, TRF_HEADS, TRF_LAYERS, TRF_FFN, TRF_DROPOUT = 64, 4, 1, 128, 0.2

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
def standardize_per_user(Xtr, Utr, *others_with_users):
    Xtr = Xtr.astype(np.float32, copy=True)
    C = Xtr.shape[2]
    global_mu = Xtr.reshape(-1, C).mean(0)
    global_sd = Xtr.reshape(-1, C).std(0) + 1e-8
    mu, sd = {}, {}
    for u in np.unique(Utr):
        Xu = Xtr[Utr==u].reshape(-1, C)
        mu[int(u)] = Xu.mean(0)
        sd[int(u)] = Xu.std(0) + 1e-8
    def _apply(X, U):
        Y = np.empty_like(X, dtype=np.float32)
        for u in np.unique(U):
            m = mu.get(int(u), global_mu)
            s = sd.get(int(u), global_sd)
            Xi = X[U==u]
            Y[U==u] = (Xi - m) / s
        return Y
    outs = [ _apply(Xtr, Utr) ]
    for (X, U) in others_with_users:
        outs.append(_apply(X, U))
    return outs

def compute_class_weights(y, neg_boost=1.0):
    classes = np.unique(y)
    cnts = np.array([(y==c).sum() for c in classes], dtype=np.float32)
    n, k = len(y), len(classes)
    w = {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, cnts)}
    if 0 in w:
        w[0] = w[0] * neg_boost
    return w

def agg_probs_window_to_file(win_probs, file_ids, mode="median", trim_q=0.10):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(win_probs, file_ids):
        d[int(f)].append(float(p))
    out = {}
    for f, ps in d.items():
        ps = np.array(ps, dtype=float)
        if mode == "median":
            out[f] = float(np.median(ps))
        elif mode == "trimmed":
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

def best_threshold(p, y, tmin, tmax, steps, metric="balanced_acc", tie_break="f1_macro", avoid_single_class=True):
    grid = np.linspace(tmin, tmax, steps)
    best_t, best_primary, best_secondary = 0.5, -1.0, -1.0
    best_pred = None
    for t in grid:
        yhat = (p > t).astype(int)
        if metric == "balanced_acc":
            primary = balanced_accuracy_score(y, yhat)
        elif metric == "f1_macro":
            primary = f1_score(y, yhat, average="macro", zero_division=0)
        else:
            primary = accuracy_score(y, yhat)
        if tie_break == "f1_macro":
            secondary = f1_score(y, yhat, average="macro", zero_division=0)
        elif tie_break == "acc":
            secondary = accuracy_score(y, yhat)
        else:
            secondary = balanced_accuracy_score(y, yhat)
        if (primary > best_primary) or (primary == best_primary and secondary > best_secondary):
            best_primary, best_secondary = primary, secondary
            best_t, best_pred = t, yhat
    if avoid_single_class and (best_pred is not None) and (len(np.unique(best_pred)) == 1):
        grid = np.linspace(tmin, tmax, steps * 2)
        for t in grid:
            yhat = (p > t).astype(int)
            if len(np.unique(yhat)) > 1:
                if metric == "balanced_acc":
                    primary = balanced_accuracy_score(y, yhat)
                elif metric == "f1_macro":
                    primary = f1_score(y, yhat, average="macro", zero_division=0)
                else:
                    primary = accuracy_score(y, yhat)
                return t, primary
        return 0.5, best_primary
    return best_t, best_primary

def _check_single_class(name, yhat):
    u = np.unique(yhat)
    if len(u) == 1:
        print(f"[WARN] {name} predicted a single class: {u[0]}")

def _clip01(x):
    return float(np.clip(x, 1e-6, 1 - 1e-6))

# ---------- Rank helper ----------
def rank_normalize(arr):
    # returns ranks in [0,1]
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    ranks = (ranks + 1.0) / (len(arr) + 1.0)   # avoid 0/1 extremes
    return ranks

# ======================= Calibration with CV =======================
def calibrate_best_of_two_cv(val_dict, te_dict, y_true_val, thr_range, steps, n_splits=CAL_CV_SPLITS, min_iso=MIN_ISO_SAMPLES):
    files_v = sorted(val_dict.keys())
    pv = np.array([val_dict[f] for f in files_v], dtype=float)
    yv = np.array([y_true_val[f] for f in files_v], dtype=int)

    files_t = sorted(te_dict.keys())
    pt = np.array([te_dict[f] for f in files_t], dtype=float)

    pv = np.clip(pv, 1e-6, 1-1e-6)
    pt = np.clip(pt, 1e-6, 1-1e-6)

    if len(np.unique(yv)) < 2 or len(pv) < 5:
        return val_dict, te_dict, "none"

    n_splits_eff = max(2, min(n_splits, len(pv)//2))
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_STATE)

    scores = {}
    for mode in ["platt", "isotonic"]:
        if mode == "isotonic" and len(pv) < min_iso:
            scores[mode] = -1.0
            continue
        oof_pred = np.zeros_like(pv, dtype=float)
        for tr, va in skf.split(pv.reshape(-1,1), yv):
            if mode == "platt":
                lr = LogisticRegression(max_iter=1000)
                lr.fit(pv[tr].reshape(-1,1), yv[tr])
                oof_pred[va] = lr.predict_proba(pv[va].reshape(-1,1))[:,1]
            else:
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(pv[tr], yv[tr].astype(float))
                oof_pred[va] = iso.predict(pv[va])
        oof_pred = np.clip(oof_pred, 1e-6, 1-1e-6)
        thr, score = best_threshold(oof_pred, yv, thr_range[0], thr_range[1], steps,
                                    metric="balanced_acc", tie_break="f1_macro")
        scores[mode] = score

    chosen = "platt" if scores.get("platt", -1) >= scores.get("isotonic", -1) else "isotonic"

    # Fit chosen on full val
    if chosen == "platt":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(pv.reshape(-1,1), yv)
        v_probs = lr.predict_proba(pv.reshape(-1,1))[:,1]
        t_probs = lr.predict_proba(pt.reshape(-1,1))[:,1]
    else:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(pv, yv.astype(float))
        v_probs = iso.predict(pv)
        t_probs = iso.predict(pt)

    v_cal = {f: _clip01(p) for f, p in zip(files_v, v_probs)}
    t_cal = {f: _clip01(p) for f, p in zip(files_t, t_probs)}
    return v_cal, t_cal, chosen

# ======================= Models =======================
class PositionalEncoding(Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def call(self, x):
        t = tf.shape(x)[1]
        i = tf.range(self.d_model, dtype=tf.float32)
        pos = tf.range(t, dtype=tf.float32)[:, None]
        angle_rates = 1.0 / tf.pow(10000.0, (2*(tf.floor(i/2.0)))/tf.cast(self.d_model, tf.float32))
        angles = pos * angle_rates[None, :]
        sines = tf.sin(angles[:, 0::2]); cosines = tf.cos(angles[:, 1::2])
        pe = tf.concat([sines, cosines], axis=-1)[None, :, :]
        return x + tf.cast(pe, x.dtype)

def build_cnn(input_shape):
    model = Sequential([
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
        Dense(64, activation='relu'),
        Dropout(DROPOUT_HEAD),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_transformer(input_shape, d_model=TRF_DMODEL, heads=TRF_HEADS, layers=TRF_LAYERS, ffn=TRF_FFN, dropout=TRF_DROPOUT):
    inp = Input(shape=input_shape)
    x = Conv1D(d_model, 1, padding='same')(inp)
    x = PositionalEncoding(d_model)(x)
    for _ in range(layers):
        x1 = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=d_model//heads, dropout=dropout)(x, x)
        x  = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x + x1)
        x2 = Sequential([Dense(ffn, activation='relu'), Dropout(dropout), Dense(d_model)])(x)
        x  = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x + x2)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def inception_module(x, nb_filters):
    t1 = Conv1D(nb_filters, 1, padding='same', activation='relu')(x)
    t2 = Conv1D(nb_filters, 3, padding='same', activation='relu')(x)
    t3 = Conv1D(nb_filters, 5, padding='same', activation='relu')(x)
    t4 = MaxPooling1D(3, strides=1, padding='same')(x)
    t4 = Conv1D(nb_filters, 1, padding='same', activation='relu')(t4)
    return tf.keras.layers.Concatenate()([t1,t2,t3,t4])

def build_inceptiontime(input_shape):
    inp = Input(shape=input_shape)
    x = inception_module(inp, 32); x = BatchNormalization()(x)
    x = inception_module(x, 32);  x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x); x = Dropout(DROPOUT_BLOCK)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x); x = Dropout(DROPOUT_HEAD)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
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

# ======================= Meta features (window-level) =======================
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

def build_meta_basic(vc, vt, vi):
    Xmeta = np.stack([vc, vt, vi], axis=1)
    mean = Xmeta.mean(1); mx = Xmeta.max(1); mn = Xmeta.min(1); var = Xmeta.var(1)
    return np.stack([vc, vt, vi, mean, mx, mn, var], axis=1)

def build_meta_window(F_ids, p_win_c, p_win_t, p_win_i, files):
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

# ======================= LR-ensemble with OOF CV =======================
def lr_fit_oof_thresh(val_mat, val_y, n_splits=LR_CV_SPLITS):
    n_splits_eff = max(2, min(n_splits, len(val_y)//2))
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(val_y), dtype=float)
    for tr, va in skf.split(val_mat, val_y):
        lr = LogisticRegression(max_iter=2000)
        lr.fit(val_mat[tr], val_y[tr])
        oof_pred[va] = lr.predict_proba(val_mat[va])[:,1]
    thr_v, score_v = best_threshold(oof_pred, val_y, THR_MIN, THR_MAX, THR_STEPS,
                                    metric="balanced_acc", tie_break="f1_macro")
    lr_final = LogisticRegression(max_iter=2000)
    lr_final.fit(val_mat, val_y)
    pos_rate = float((oof_pred > thr_v).mean())
    return lr_final, thr_v, score_v, pos_rate

# ======================= Test-time POS rate adaptation =======================
def adapt_threshold_to_pos_rate(p_val, thr_v, p_test, clip=ADAPT_POS_CLIP):
    r_val = float((p_val > thr_v).mean())
    r_tgt = float(np.clip(r_val, clip[0], clip[1]))
    # threshold is the (1 - r_tgt) quantile of test scores
    q = 1.0 - r_tgt
    q = float(np.clip(q, 0.0+1e-6, 1.0-1e-6))
    thr_t = float(np.quantile(p_test, q))
    # guard: avoid single-class by nudging inside the support
    lo, hi = float(np.min(p_test)), float(np.max(p_test))
    eps = 1e-6
    thr_t = float(np.clip(thr_t, lo+eps, hi-eps))
    return thr_t, r_tgt

# ======================= Weighted & Rank ensembles =======================
WEIGHT_CANDIDATES = [
    (0.2,0.5,0.3), (0.25,0.5,0.25), (0.3,0.4,0.3), (0.3,0.5,0.2),
    (0.4,0.4,0.2), (0.4,0.3,0.3), (0.2,0.4,0.4), (0.33,0.34,0.33),
]

def weighted_ensemble_select(vc, vt, vi, vy):
    best = (-1.0, (1/3,1/3,1/3), 0.5, 0.0)  # score, weights, thr, pos_rate
    for (wc, wt, wi) in WEIGHT_CANDIDATES:
        v_ens = wc*vc + wt*vt + wi*vi
        thr, score = best_threshold(v_ens, vy, THR_MIN, THR_MAX, THR_STEPS,
                                    metric="balanced_acc", tie_break="f1_macro")
        pos_rate = float((v_ens > thr).mean())
        if score > best[0]:
            best = (score, (wc, wt, wi), thr, pos_rate)
    return best  # score, weights, thr, pos_rate

def rank_average(arr_list):
    # arr_list: list of arrays shape (n_files,)
    ranks = [rank_normalize(a) for a in arr_list]
    mean_rank = np.mean(np.stack(ranks, axis=1), axis=1)
    return mean_rank

# ======================= Main CV Loop =======================
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
    val_files = choose_files_for_validation_stratified(train_files, y_file, VAL_FILE_RATIO,
                                                       seed=RANDOM_STATE + fold)
    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # per-user standardization
    Utr_all = Gw[tr_idx]; Ute_all = Gw[te_idx]
    Utr_in  = Utr_all[tr_mask]; Uva_in = Utr_all[va_mask]
    Xtr_in, Xva_in, Xte_std = standardize_per_user(
        Xtr_in, Utr_in,
        (Xva_in, Uva_in),
        (Xte,    Ute_all)
    )
    Xte = Xte_std

    # class weights
    classes = np.unique(Ytr_in)
    cnts = np.array([(Ytr_in==c).sum() for c in classes], dtype=float)
    n, k = len(Ytr_in), len(classes)
    cw = {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, cnts)}
    if 0 in cw: cw[0] *= NEG_BOOST
    print("[CLASS WEIGHTS]", cw, flush=True)

    # Base models (window-level)
    def build_inceptiontime(input_shape):
        inp = Input(shape=input_shape)
        x = inception_module(inp, 32); x = BatchNormalization()(x)
        x = inception_module(x, 32);  x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x); x = Dropout(DROPOUT_BLOCK)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x); x = Dropout(DROPOUT_HEAD)(x)
        out = Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    val_cnn, te_cnn = fit_seed_ensemble(build_cnn,          Xtr_in, Ytr_in, Xva_in, Xte, cw, SEEDS)
    val_trf, te_trf = fit_seed_ensemble(lambda s: build_transformer(s), Xtr_in, Ytr_in, Xva_in, Xte, cw, SEEDS)
    val_inc, te_inc = fit_seed_ensemble(build_inceptiontime, Xtr_in, Ytr_in, Xva_in, Xte, cw, SEEDS)

    # Aggregation sweep + calibration per model (CV-based)
    agg_candidates = []
    for mode in AGG_MODE_CAND:
        if mode == "trimmed":
            for tq in TRIM_Q_CAND:
                agg_candidates.append((mode, tq))
        else:
            agg_candidates.append((mode, 0.10))

    best_val_score_overall = -1
    best_cfg = None
    best_data = None

    for (mode, tq) in agg_candidates:
        # file-level (uncalibrated)
        v_file_cnn = agg_probs_window_to_file(val_cnn, Fva_in, mode=mode, trim_q=tq)
        v_file_trf = agg_probs_window_to_file(val_trf, Fva_in, mode=mode, trim_q=tq)
        v_file_inc = agg_probs_window_to_file(val_inc, Fva_in, mode=mode, trim_q=tq)

        t_file_cnn = agg_probs_window_to_file(te_cnn, Fte, mode=mode, trim_q=tq)
        t_file_trf = agg_probs_window_to_file(te_trf, Fte, mode=mode, trim_q=tq)
        t_file_inc = agg_probs_window_to_file(te_inc, Fte, mode=mode, trim_q=tq)

        v_files = sorted(set(v_file_cnn.keys()) & set(v_file_trf.keys()) & set(v_file_inc.keys()))
        yv_map = {int(f): int(Y[f]) for f in v_files}

        # calibration with CV per model
        v_cal_cnn, t_cal_cnn, m_cnn = calibrate_best_of_two_cv(v_file_cnn, t_file_cnn, yv_map,
                                                               thr_range=(THR_MIN,THR_MAX), steps=THR_STEPS)
        v_cal_trf, t_cal_trf, m_trf = calibrate_best_of_two_cv(v_file_trf, t_file_trf, yv_map,
                                                               thr_range=(THR_MIN,THR_MAX), steps=THR_STEPS)
        v_cal_inc, t_cal_inc, m_inc = calibrate_best_of_two_cv(v_file_inc, t_file_inc, yv_map,
                                                               thr_range=(THR_MIN,THR_MAX), steps=THR_STEPS)

        # build validation arrays
        v_y = np.array([Y[f] for f in v_files], dtype=int)
        v_c = np.array([v_cal_cnn[f] for f in v_files]); v_t = np.array([v_cal_trf[f] for f in v_files]); v_i = np.array([v_cal_inc[f] for f in v_files])

        # --- M1: LR stacking with meta features (incl. window meta)
        Xmeta_v_basic = build_meta_basic(v_c, v_t, v_i)
        Xmeta_v_win   = build_meta_window(Fva_in, val_cnn, val_trf, val_inc, v_files)
        V_meta = np.concatenate([Xmeta_v_basic, Xmeta_v_win], axis=1)

        lr_stk, thr_v_m1, score_v_m1, posrate_m1 = lr_fit_oof_thresh(V_meta, v_y, n_splits=LR_CV_SPLITS)
        # cache val predictions for adaptation
        p_val_m1 = lr_stk.predict_proba(V_meta)[:,1]

        # --- M2: Weighted average
        score_v_m2, best_w, thr_v_m2, posrate_m2 = weighted_ensemble_select(v_c, v_t, v_i, v_y)
        v_ens_m2 = best_w[0]*v_c + best_w[1]*v_t + best_w[2]*v_i
        p_val_m2 = v_ens_m2

        # --- M3: Rank-average
        v_rank = rank_average([v_c, v_t, v_i])
        thr_v_m3, score_v_m3 = best_threshold(v_rank, v_y, THR_MIN, THR_MAX, THR_STEPS,
                                              metric="balanced_acc", tie_break="f1_macro")
        posrate_m3 = float((v_rank > thr_v_m3).mean())
        p_val_m3   = v_rank

        # pick best head by val score
        heads = [("m1_stack", score_v_m1), ("m2_weighted", score_v_m2), ("m3_rank", score_v_m3)]
        heads.sort(key=lambda x: x[1], reverse=True)
        best_head = heads[0][0]

        if heads[0][1] > best_val_score_overall:
            best_val_score_overall = heads[0][1]
            best_cfg = {
                "agg_mode": mode, "trim_q": float(tq),
                "cal": {"cnn": m_cnn, "trf": m_trf, "inc": m_inc},
                "picked_head": best_head
            }
            best_data = {
                "v_files": v_files, "v_y": v_y,
                "v_c": v_c, "v_t": v_t, "v_i": v_i,
                "p_val_m1": p_val_m1, "thr_v_m1": float(thr_v_m1), "lr_stk": lr_stk,
                "p_val_m2": p_val_m2, "thr_v_m2": float(thr_v_m2), "w_m2": best_w,
                "p_val_m3": p_val_m3, "thr_v_m3": float(thr_v_m3)
            }
            best_test_sets = (t_cal_cnn, t_cal_trf, t_cal_inc)

    # ----- Apply best config on TEST with POS-RATE ADAPTATION -----
    v_files = best_data["v_files"]; v_y = best_data["v_y"]
    v_c = best_data["v_c"]; v_t = best_data["v_t"]; v_i = best_data["v_i"]
    t_cal_cnn, t_cal_trf, t_cal_inc = best_test_sets

    t_files = sorted(set(t_cal_cnn.keys()) & set(t_cal_trf.keys()) & set(t_cal_inc.keys()))
    t_y = np.array([Y[f] for f in t_files], dtype=int)
    t_c = np.array([t_cal_cnn[f] for f in t_files]); t_t = np.array([t_cal_trf[f] for f in t_files]); t_i = np.array([t_cal_inc[f] for f in t_files])

    picked = best_cfg["picked_head"]

    if picked == "m1_stack":
        # build test meta
        Xmeta_t_basic = build_meta_basic(t_c, t_t, t_i)
        # window meta on TEST نمی‌توانیم بسازیم چون به window-level تست نیاز داریم؟ داریم: te_* و Fte موجود‌اند.
        # ولی در این محدوده بعد از حلقه sweep، te_* و Fte هنوز در scope هستند.
        # پس می‌سازیم:
        # NOTE: te_cnn/trf/inc و Fte از بیرون حلقه sweep قابل دسترسی‌اند.
        Xmeta_t_win = build_meta_window(Fte, te_cnn, te_trf, te_inc, t_files)
        T_meta = np.concatenate([Xmeta_t_basic, Xmeta_t_win], axis=1)

        lr_stk = best_data["lr_stk"]
        p_test = lr_stk.predict_proba(T_meta)[:,1]
        thr_v  = best_data["thr_v_m1"]
        thr_t, r_tgt = adapt_threshold_to_pos_rate(best_data["p_val_m1"], thr_v, p_test, clip=ADAPT_POS_CLIP)

    elif picked == "m2_weighted":
        w = best_data["w_m2"]; p_test = w[0]*t_c + w[1]*t_t + w[2]*t_i
        thr_v  = best_data["thr_v_m2"]
        thr_t, r_tgt = adapt_threshold_to_pos_rate(best_data["p_val_m2"], thr_v, p_test, clip=ADAPT_POS_CLIP)

    else:  # m3_rank
        p_test = rank_average([t_c, t_t, t_i])
        thr_v  = best_data["thr_v_m3"]
        thr_t, r_tgt = adapt_threshold_to_pos_rate(best_data["p_val_m3"], thr_v, p_test, clip=ADAPT_POS_CLIP)

    yhat = (p_test > thr_t).astype(int)
    _check_single_class(f"{picked}", yhat)

    acc  = accuracy_score(t_y, yhat)
    prec = precision_score(t_y, yhat, zero_division=0)
    rec  = recall_score(t_y, yhat, zero_division=0)
    f1   = f1_score(t_y, yhat, zero_division=0)
    bal  = balanced_accuracy_score(t_y, yhat)

    print(f"[FOLD {fold}] Best Config → agg={best_cfg['agg_mode']} trim_q={best_cfg['trim_q']:.2f} | cal={best_cfg['cal']} | Val(best_bal)={best_val_score_overall:.3f} | Head={picked}")
    print(f"[FOLD {fold}] Test: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} BalAcc={bal:.3f} | thr_v={thr_v:.3f} → thr_t={thr_t:.3f} (target pos≈{r_tgt:.2f})")

    rows.append({
        "fold":fold, "mode":picked,
        "acc":acc,"precision":prec,"recall":rec,"f1":f1,"bal_acc":bal,
        "thr_t":float(thr_t),
        "agg_mode": best_cfg['agg_mode'], "trim_q": float(best_cfg['trim_q']),
        "cal_cnn": best_cfg['cal']['cnn'], "cal_trf": best_cfg['cal']['trf'], "cal_inc": best_cfg['cal']['inc']
    })

    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

def summarize(df):
    mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
    return {
        "acc_mean": float(mean.get("acc", np.nan)), "acc_std": float(std.get("acc", np.nan)),
        "prec_mean": float(mean.get("precision", np.nan)), "prec_std": float(std.get("precision", np.nan)),
        "rec_mean": float(mean.get("recall", np.nan)), "rec_std": float(std.get("recall", np.nan)),
        "f1_mean": float(mean.get("f1", np.nan)), "f1_std": float(std.get("f1", np.nan)),
        "bal_acc_mean": float(mean.get("bal_acc", np.nan)), "bal_acc_std": float(std.get("bal_acc", np.nan)),
    }

sum_all = {
    "summary": summarize(df),
    "config": {
        "WIN":WIN,"STR":STR,"Seeds":SEEDS,"TopK":TOPK_K,"NEG_BOOST":NEG_BOOST,
        "VAL_FILE_RATIO":VAL_FILE_RATIO,
        "THR_RANGE":[THR_MIN,THR_MAX,THR_STEPS],
        "Transformer":{"d_model":TRF_DMODEL,"heads":TRF_HEADS,"layers":TRF_LAYERS,"ffn":TRF_FFN,"dropout":TRF_DROPOUT},
        "agg_candidates": {"modes": AGG_MODE_CAND, "trim_q": TRIM_Q_CAND},
        "calibration_cv_splits": CAL_CV_SPLITS,
        "lr_cv_splits": LR_CV_SPLITS,
        "standardization": "per-user",
        "threshold_metric": "balanced_acc (OOF/tie=f1_macro)",
        "test_threshold": "pos-rate adaptation (quantile) with clip",
        "heads": ["m1_stack_meta", "m2_weighted", "m3_rank_avg"]
    }
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(sum_all, f, indent=2, ensure_ascii=False)

print("\n[SUMMARY]")
print(json.dumps(sum_all, indent=2, ensure_ascii=False))
print("Saved to:", RESULT_DIR, flush=True)
