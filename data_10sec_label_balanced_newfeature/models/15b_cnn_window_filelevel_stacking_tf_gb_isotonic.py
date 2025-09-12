# 15b_cnn_window_filelevel_stacking_tf_gb_isotonic.py
import os, math, random, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ---------------- Paths ----------------
DATA_DIR   = os.path.join("..","data")
RESULT_DIR = os.path.join("..","results","15b_cnn_window_filelevel_stacking_tf_gb_isotonic")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR,"X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR,"Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR,"G8.npy")).astype(int)
meta = pd.read_csv(os.path.join(DATA_DIR,"meta_features_8cols.csv"))  # website, task,...

assert len(meta)==len(X)
sites = meta["website"].astype(str).str.lower().values
tasks = meta["task"].astype(int).values

tab_path = os.path.join(DATA_DIR,"X_tabular.npy")
USE_TAB = os.path.exists(tab_path)
if USE_TAB:
    X_tab = np.load(tab_path)  # (N,96)

N,T,C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}, USE_TAB={USE_TAB}", flush=True)

# ---------------- Config ----------------
WIN, STR     = 400, 200
SEEDS        = [7,17,23]
EPOCHS       = 80
BATCH_SIZE   = 16
LR_INIT      = 1e-3
L2_REG       = 1e-5
DROP_BLOCK   = 0.30
DROP_HEAD    = 0.35
RANDOM_STATE = 42
N_SPLITS     = 6

THR_MIN, THR_MAX, THR_STEPS = 0.35, 0.75, 41
VAL_FILE_RATIO = 0.30  # برای انتخاب آستانه فایل‌سطح (و فیت ایزوتونیک)

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

# ---------------- utils ----------------
def make_windows(X, Y, G, win=WIN, stride=STR):
    Xw,Yw,Gw,Fw = [],[],[],[]
    for i in range(X.shape[0]):
        xi = X[i]
        for s in range(0, T-win+1, stride):
            Xw.append(xi[s:s+win,:]); Yw.append(Y[i]); Gw.append(G[i]); Fw.append(i)
    return np.asarray(Xw, np.float32), np.asarray(Yw, int), np.asarray(Gw, int), np.asarray(Fw, int)

def standardize_per_fold(train_arr, *others):
    t_len, c = train_arr.shape[1], train_arr.shape[2]
    sc = StandardScaler().fit(train_arr.reshape(-1, c))
    outs=[]
    for arr in (train_arr,)+others:
        A = sc.transform(arr.reshape(-1, c)).reshape(arr.shape[0], t_len, c)
        outs.append(A)
    return outs

def compute_class_weights(y):
    classes = np.unique(y)
    cnts = np.array([(y==c).sum() for c in classes], dtype=np.float32)
    n,k = len(y), len(classes)
    return {int(c): float(n/(k*cnt)) for c,cnt in zip(classes,cnts)}

def build_cnn(input_shape):
    m = Sequential([
        tf.keras.Input(shape=input_shape),
        Conv1D(48,7,padding='same',kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROP_BLOCK),

        Conv1D(64,5,padding='same',kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROP_BLOCK),

        Conv1D(64,3,padding='same',kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROP_BLOCK),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(DROP_HEAD),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def collect_probs_by_file(probs, file_ids):
    from collections import defaultdict
    d=defaultdict(list)
    for p,f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    return d

def choose_val_files_stratified(file_ids, y_file, ratio, seed):
    rng = np.random.default_rng(seed)
    ids = np.unique(file_ids)
    ids0 = [i for i in ids if y_file[i]==0]
    ids1 = [i for i in ids if y_file[i]==1]
    n_val = max(1, int(math.ceil(len(ids)*ratio)))
    n1 = max(1, int(round(n_val * len(ids1)/max(1,len(ids)))))
    n0 = max(1, n_val - n1)
    sel = set()
    if ids0: sel |= set(rng.choice(ids0, size=min(n0,len(ids0)), replace=False).tolist())
    if ids1: sel |= set(rng.choice(ids1, size=min(n1,len(ids1)), replace=False).tolist())
    # تکمیل اگر کم شد
    rest = [i for i in ids if i not in sel]
    while len(sel)<n_val and rest:
        sel.add(rest.pop())
    return sel

def grid_thr_for_accuracy(p, y, tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS):
    grid = np.linspace(tmin, tmax, steps)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        yhat = (p>t).astype(int)
        acc = accuracy_score(y, yhat)
        if acc>best_acc: best_acc, best_t = acc, t
    return best_t, best_acc

def add_file_features(prob_list):
    import numpy as np, math
    arr = np.array(prob_list, dtype=float)
    if arr.size==0:
        arr = np.array([0.0])
    feats = {}
    # پایه‌ای
    feats["n_win"] = len(arr)
    feats["p_mean"] = float(np.mean(arr))
    feats["p_std"]  = float(np.std(arr))
    feats["p_min"]  = float(np.min(arr))
    feats["p_q25"]  = float(np.percentile(arr,25))
    feats["p_med"]  = float(np.median(arr))
    feats["p_q75"]  = float(np.percentile(arr,75))
    feats["p_max"]  = float(np.max(arr))
    # proportion thresholds
    for th in [0.55,0.60,0.65,0.70,0.75]:
        feats[f"prop_gt_{str(th).replace('.','p')}"] = float(np.mean(arr>th))
    # top-k means
    for k in [3,5]:
        topk = np.sort(arr)[-k:] if len(arr)>=k else arr
        feats[f"top{k}_mean"] = float(np.mean(topk))
    # skew & kurt (محافظه‌کارانه)
    m = feats["p_mean"]; s = feats["p_std"]+1e-9
    z = (arr-m)/s
    feats["skew"] = float(np.mean(z**3))
    feats["kurt"] = float(np.mean(z**4)-3.0)
    # entropy over binned probs
    bins = np.histogram(arr, bins=[0,0.5,0.6,0.7,0.8,0.9,1.0])[0] + 1e-9
    pbin = bins/np.sum(bins)
    feats["entropy_bins"] = float(-np.sum(pbin*np.log(pbin)))
    return feats

# ---------------- run ----------------
Xw, Yw, Gw, Fw = make_windows(X,Y,G,WIN,STR)
print(f"[WINDOWS] {Xw.shape} {Yw.shape} {Gw.shape} {Fw.shape}", flush=True)

gkf = GroupKFold(n_splits=N_SPLITS)
rows=[]; fold=1

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)
    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    Xtr, Xte = standardize_per_fold(Xtr, Xte)
    cw = compute_class_weights(Ytr)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # seed-ensemble
    te_win_probs_list=[]; tr_win_probs_list=[]
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_cnn((Xtr.shape[1], Xtr.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
        model.fit(Xtr, Ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.1, class_weight=cw, verbose=0, callbacks=[es, rlr])

        tr_win_probs_list.append(model.predict(Xtr, verbose=0).ravel())
        te_win_probs_list.append(model.predict(Xte, verbose=0).ravel())

    tr_win_probs = np.mean(tr_win_probs_list, axis=0)
    te_win_probs = np.mean(te_win_probs_list, axis=0)

    # فایل‌های ولیدیشن فایل‌سطح (برای ایزوتونیک + انتخاب آستانه نهایی)
    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}
    val_files = choose_val_files_stratified(train_files, y_file, VAL_FILE_RATIO, seed=RANDOM_STATE+fold)

    # جمع‌آوری به ازای فایل
    def group_by_file(probs, files):
        from collections import defaultdict
        d=defaultdict(list)
        for p,f in zip(probs, files):
            d[int(f)].append(float(p))
        return d

    tr_dict = group_by_file(tr_win_probs, Ftr)
    te_dict = group_by_file(te_win_probs, Fte)

    # کالیبراسیون ایزوتونیک در سطح «میانگین فایل»
    # ابتدا میانگین خام و سپس فیت ایزوتونیک روی فایل‌های ولیدیشن
    v_files = sorted(list(val_files))
    if len(v_files)>=2 and len(np.unique([Y[f] for f in v_files]))==2:
        v_raw = np.array([np.mean(tr_dict[f]) for f in v_files])
        v_lab = np.array([Y[f] for f in v_files])
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(v_raw, v_lab)
        def calib_file_prob_list(d):
            # هر فایل: هر پنجره را کالیبره نمی‌کنیم؛ بلکه میانگین پنجره‌ها را کالیبره و سپس برای فیچرها از دو نسخه استفاده می‌کنیم
            out = {}
            for f,lst in d.items():
                mean_raw = float(np.mean(lst))
                mean_cal = float(ir.transform([mean_raw])[0])
                # نسخه‌ی «prob کالیبره‌شده‌ی فایل» را به همه‌ی پنجره‌ها نسبت نمی‌دهیم؛
                # فقط به‌عنوان یک فیچر اضافه خواهیم کرد.
                out[int(f)] = (lst, mean_cal)
            return out
        tr_aug = calib_file_prob_list(tr_dict)
        te_aug = calib_file_prob_list(te_dict)
    else:
        # بدون ایزوتونیک
        tr_aug = {int(f):(lst, float(np.mean(lst))) for f,lst in tr_dict.items()}
        te_aug = {int(f):(lst, float(np.mean(lst))) for f,lst in te_dict.items()}

    # ساخت فیچرهای فایل‌سطح
    def build_filelevel_df(aug_dict):
        rows=[]
        for f,(lst,mean_cal) in aug_dict.items():
            feats = add_file_features(lst)           # آمارهای توزیع پنجره‌ها (خام)
            feats["mean_cal"] = float(mean_cal)      # میانگین کالیبره شده
            feats["site"] = sites[f]
            feats["task"] = int(tasks[f])
            if USE_TAB:
                # 96 ویژگی آماری (قبلاً ساخته‌ای) را هم اضافه کن
                for j,val in enumerate(X_tab[f].tolist()):
                    feats[f"tab_{j}"] = float(val)
            feats["y"] = int(Y[f])
            feats["file"] = int(f)
            rows.append(feats)
        df = pd.DataFrame(rows).set_index("file").sort_index()
        return df

    tr_df = build_filelevel_df(tr_aug)
    te_df = build_filelevel_df(te_aug)

    # وان‌هات site/task
    enc_site = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc_task = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    site_tr_oh = enc_site.fit_transform(tr_df[["site"]])
    site_te_oh = enc_site.transform(te_df[["site"]])
    task_tr_oh = enc_task.fit_transform(tr_df[["task"]])
    task_te_oh = enc_task.transform(te_df[["task"]])

    # جدا کردن y و ستون‌های پیوسته
    y_tr = tr_df["y"].values; y_te = te_df["y"].values
    drop_cols = ["site","task","y"]
    cont_tr = tr_df.drop(columns=drop_cols).values
    cont_te = te_df.drop(columns=drop_cols).values

    # ترکیب ویژگی‌ها
    Xtr_f = np.hstack([cont_tr, site_tr_oh, task_tr_oh])
    Xte_f = np.hstack([cont_te, site_te_oh, task_te_oh])

    # مدل فایل‌سطح: Gradient Boosting (بدون نرمال‌سازی)
    gb = HistGradientBoostingClassifier(
        max_depth=3, max_iter=300, learning_rate=0.06,
        l2_regularization=1e-3, early_stopping=True, random_state=RANDOM_STATE+fold
    )
    gb.fit(Xtr_f, y_tr)
    te_prob = gb.predict_proba(Xte_f)[:,1]

    # انتخاب آستانه برای بیشینه‌سازی Accuracy (روی بخشی از train که val_files بود)
    # توجه: چون ایزوتونیک را روی val_files فیت کردیم، برای threshold هم از همان فایل‌ها استفاده می‌کنیم.
    v_probs = []
    v_true  = []
    for f in v_files:
        # احتمال فایل‌سطح برای فایل‌های val از روی gb اما باید از tr_df/encoderها عبور کنیم.
        # ترفند ساده: نمونه‌ی فایل‌سطح val را از tr_df انتخاب و همان pipeline را اعمال کنیم:
        row = tr_df.loc[f:f]  # DataFrame تک‌سطر
        site_v = enc_site.transform(row[["site"]])
        task_v = enc_task.transform(row[["task"]])
        cont_v = row.drop(columns=["site","task","y"]).values
        Xv = np.hstack([cont_v, site_v, task_v])
        v_probs.append(gb.predict_proba(Xv)[:,1][0])
        v_true.append(int(row["y"].values[0]))
    v_probs = np.array(v_probs); v_true = np.array(v_true, int)

    thr, acc_val = grid_thr_for_accuracy(v_probs, v_true, THR_MIN, THR_MAX, THR_STEPS)
    print(f"[FOLD {fold}] best_thr={thr:.3f} | Acc(val-files)={acc_val:.3f}")

    y_pred = (te_prob > thr).astype(int)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)

    rows.append({"fold":fold, "thr":float(thr),
                 "acc":acc,"precision":prec,"recall":rec,"f1":f1,
                 "n_test_files":len(y_te)})
    print(f"[FOLD {fold}] FILE-LEVEL GB → Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
    fold += 1

# -------- Summary --------
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR,"metrics_file_level.csv"), index=False)
mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = [
    f"WIN={WIN}, STR={STR}, Seeds={SEEDS}, USE_TAB={USE_TAB}, VAL_FILE_RATIO={VAL_FILE_RATIO}",
    "File-level thresholds: "+", ".join([f"{t:.3f}" for t in df['thr'].tolist()]),
    "Mean ± STD (file-level, GB stacking)",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall: {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1: {mean['f1']:.4f} ± {std['f1']:.4f}",
]
with open(os.path.join(RESULT_DIR,"summary.txt"),"w",encoding="utf-8") as f:
    f.write("\n".join(summary))
print("\n".join(summary), flush=True)
print("Saved to:", RESULT_DIR, flush=True)
