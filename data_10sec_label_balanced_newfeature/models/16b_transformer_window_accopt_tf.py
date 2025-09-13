# 16b_transformer_window_accopt_tf.py
# Transformer encoder for gaze deltas
# windowing + per-fold standardization + class_weight (NEG_BOOST)
# seed-ensemble + TopK snapshots + file-level aggregation (trimmed) + acc-opt threshold

import os, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Dense, Dropout, LayerNormalization,
                                     GlobalAveragePooling1D, GlobalMaxPooling1D,
                                     Concatenate, Input, MultiHeadAttention)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============== Paths ==============
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "16b_transformer_window_accopt_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ============== Config (tuned) ==============
WIN, STR        = 500, 200
N_SPLITS        = 6
EPOCHS          = 100
BATCH_SIZE      = 16
LR_INIT         = 1e-3
RANDOM_STATE    = 42
VAL_FILE_RATIO  = 0.35
THR_MIN, THR_MAX = 0.40, 0.80
THR_STEPS       = 41
SEEDS           = [7, 17, 23, 31, 57]  # stronger ensemble
TOPK_K          = 5
NEG_BOOST       = 1.4
AGG_MODE        = "trimmed"            # robust aggregation

# Transformer hyperparams
D_MODEL         = 64
N_HEADS         = 4        # must divide D_MODEL
FFN_DIM         = 128
N_LAYERS        = 2
DROPOUT         = 0.2
L2_REG          = 1e-5

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

# ============== Windowing ==============
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

# ============== Utils ==============
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

def _agg(arr, mode):
    if mode == "median":
        return np.median(arr) if len(arr) else 0.0
    elif mode == "trimmed":
        if len(arr) == 0: return 0.0
        lo, hi = np.percentile(arr, [10, 90])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        return np.mean(trimmed) if len(trimmed) else np.mean(arr)
    else:
        return np.mean(arr) if len(arr) else 0.0

def agg_probs_file_level(probs, file_ids, mode=AGG_MODE):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    return {f: float(_agg(np.array(ps, dtype=float), mode)) for f, ps in d.items()}

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

def choose_threshold_for_accuracy(val_file_probs, val_file_true, tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS):
    grid = np.linspace(tmin, tmax, steps)
    files = list(val_file_probs.keys())
    y_true = np.array([val_file_true[f] for f in files], dtype=int)
    p_vec  = np.array([val_file_probs[f] for f in files], dtype=float)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        y_hat = (p_vec > t).astype(int)
        acc = accuracy_score(y_true, y_hat)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc

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

# ============== Transformer Blocks ==============
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, x):
        # x: (B, T, D) with D == d_model
        dtype = x.dtype
        T = tf.shape(x)[1]
        d_model = self.d_model

        pos = tf.cast(tf.range(T), dtype)[:, None]  # (T,1)
        div_term = tf.exp(
            tf.cast(tf.range(0, d_model, 2), dtype) * (-tf.math.log(10000.0) / tf.cast(d_model, dtype))
        )  # (d_model/2,)

        sin_part = tf.sin(pos * div_term)  # (T, d_model/2)
        cos_part = tf.cos(pos * div_term)  # (T, d_model/2)

        pe = tf.reshape(tf.stack([sin_part, cos_part], axis=-1), (T, -1))  # (T,d_model)
        pe = tf.expand_dims(pe, 0)  # (1,T,d_model)
        return x + pe

    def compute_output_shape(self, input_shape):
        return input_shape

def transformer_encoder(x, d_model, num_heads, ffn_dim, dropout):
    # x: (B, T, d_model)
    attn_output = MultiHeadAttention(num_heads=num_heads,
                                     key_dim=d_model//num_heads,
                                     dropout=dropout)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = tf.keras.Sequential([
        Dense(ffn_dim, activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
        Dropout(dropout),
        Dense(d_model, kernel_regularizer=regularizers.l2(L2_REG)),
    ])
    x = LayerNormalization(epsilon=1e-6)(x + ffn(x))
    return x

def build_model(input_shape):
    inp = Input(shape=input_shape)          # (T, C)
    # project channels -> d_model
    x = Dense(D_MODEL, kernel_regularizer=regularizers.l2(L2_REG))(inp)
    x = PositionalEncoding(D_MODEL)(x)

    for _ in range(N_LAYERS):
        x = transformer_encoder(x, D_MODEL, N_HEADS, FFN_DIM, DROPOUT)

    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    h = Concatenate()([gap, gmp])

    h = Dense(64, activation='relu')(h)
    h = Dropout(0.35)(h)
    out = Dense(1, activation='sigmoid')(h)

    model = Model(inp, out)
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============== GroupKFold ==============
gkf = GroupKFold(n_splits=N_SPLITS)
rows, fold = [], 1

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

    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    val_win_probs_list, te_win_probs_list = [], []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_model((Xtr_in.shape[1], Xtr_in.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=14, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_K)

        model.fit(Xtr_in, Ytr_in,
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(Xva_in, Yva_in),
                  class_weight=cw, verbose=0,
                  callbacks=[es, rlr, topk])

        topk.set_topk_weights()

        val_win_probs_list.append(model.predict(Xva_in, verbose=0).ravel())
        te_win_probs_list.append(model.predict(Xte,   verbose=0).ravel())

    val_win_probs = np.mean(val_win_probs_list, axis=0)
    te_win_probs  = np.mean(te_win_probs_list,  axis=0)

    # file-level aggregation
    def agg(probs, fids):
        from collections import defaultdict
        d = defaultdict(list)
        for p, f in zip(probs, fids):
            d[int(f)].append(float(p))
        return {f: float(_agg(np.array(ps, dtype=float), AGG_MODE)) for f, ps in d.items()}

    val_file_probs = agg(val_win_probs, Fva_in)
    te_file_probs  = agg(te_win_probs,  Fte)

    val_file_true  = {int(fi): int(Y[fi]) for fi in val_file_probs.keys()}
    thr_file, acc_val = choose_threshold_for_accuracy(val_file_probs, val_file_true,
                                                      tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS)
    print(f"[FOLD {fold}] thr_file={thr_file:.3f} | Acc(val-files)={acc_val:.3f}")

    te_files_sorted = sorted(te_file_probs.keys())
    y_true = np.array([int(Y[f]) for f in te_files_sorted], dtype=int)
    p_mean = np.array([te_file_probs[f] for f in te_files_sorted], dtype=float)
    y_pred = (p_mean > thr_file).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    rows.append({"fold":fold, "thr":float(thr_file),
                 "acc":acc, "precision":prec, "recall":rec, "f1":f1})
    fold += 1

# ============== Summary & Save ==============
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = [
    f"WIN={WIN}, STR={STR}, Seeds={SEEDS}, TopK={TOPK_K}, NEG_BOOST={NEG_BOOST}, AGG_MODE={AGG_MODE}",
    f"Transformer: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, ffn={FFN_DIM}, dropout={DROPOUT}",
    "Mean ± STD (file-level)",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall: {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1: {mean['f1']:.4f} ± {std['f1']:.4f}",
]
with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))
print("\n".join(summary), flush=True)
print("Saved to:", RESULT_DIR, flush=True)
