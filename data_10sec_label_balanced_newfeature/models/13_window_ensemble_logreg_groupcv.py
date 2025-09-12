# 13_window_ensemble_logreg_groupcv.py
# آموزش LogisticRegression روی فیچرهای پنجره‌ای و ارزیابی:
#   - GroupKFold با گروه=user روی پنجره‌ها
#   - تبدیل پیش‌بینی پنجره‌ها به پیش‌بینی سطح فایل (میانگینِ احتمالات → آستانه 0.5)
import os, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = os.path.join("..","data","windowed")

X = np.load(os.path.join(DATA_DIR,"X_win_tab.npy"))
y = np.load(os.path.join(DATA_DIR,"y_win.npy")).astype(int)
g = np.load(os.path.join(DATA_DIR,"g_win.npy")).astype(int)
fidx = np.load(os.path.join(DATA_DIR,"file_idx.npy")).astype(int)

print("[INFO]", X.shape, y.shape, g.shape, fidx.shape)

gkf = GroupKFold(n_splits=6)
accs_file, f1s_file = [], []

for tr, te in gkf.split(X, y, groups=g):
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    f_tr, f_te = fidx[tr], fidx[te]

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(Xtr, ytr)

    # پیش‌بینی در سطح پنجره
    prob_win = clf.predict_proba(Xte)[:,1]

    # تجمیع به سطح فایل: میانگین احتمال‌های پنجره‌های هر فایل
    from collections import defaultdict
    agg = defaultdict(list)
    for p, fi in zip(prob_win, f_te):
        agg[fi].append(p)

    y_true_file, y_pred_file = [], []
    for fi, plist in agg.items():
        p_mean = float(np.mean(plist))
        y_pred_file.append(1 if p_mean > 0.5 else 0)
        # y واقعی فایل مادر را از هر نمونه پنجره‌ای متناظر می‌گیریم (همه یکی‌اند)
        y_true_file.append(int(y[y==y[fidx==fi][0]][0]))  # ساده‌سازی

    accs_file.append(accuracy_score(y_true_file, y_pred_file))
    f1s_file.append(f1_score(y_true_file, y_pred_file, zero_division=0))

print(f"[FILE-LEVEL] Acc mean={np.mean(accs_file):.3f} ± {np.std(accs_file):.3f} | "
      f"F1 mean={np.mean(f1s_file):.3f} ± {np.std(f1s_file):.3f}")
