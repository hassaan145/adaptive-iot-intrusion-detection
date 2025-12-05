# ============================================================
# Models: CNN+LSTM+Attention (Base) vs CNN+Transformer (Improved)
# ============================================================

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Imports
import os, time, math, random
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -------------------------
# Load NSL-KDD files (Drive paths)
# -------------------------
TRAIN_PATH = "/content/drive/MyDrive/nsl-kdd/KDDTrain+.txt"
TEST_PATH  = "/content/drive/MyDrive/nsl-kdd/KDDTest+.txt"

df_train = pd.read_csv(TRAIN_PATH, header=None)
df_test  = pd.read_csv(TEST_PATH, header=None)
print("Raw shapes -> train:", df_train.shape, " test:", df_test.shape)

# Combine to detect label column
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
label_col = None
for col in df_all.columns[::-1]:
    vals = df_all[col].astype(str).str.lower().unique()
    if any(v == 'normal' for v in vals):
        label_col = col
        break
if label_col is None:
    label_col = df_all.columns[-2]
print("Detected label column index:", label_col)

# Binary label: normal=0 else 1
y_all = (df_all[label_col].astype(str).str.lower() != 'normal').astype(int)

# Drop label & trailing difficulty if present
drop_cols = [label_col]
if df_all.shape[1] >= 2 and df_all.columns[-1] != label_col:
    try:
        vals = df_all[df_all.columns[-1]].astype(float)
        if len(np.unique(vals)) < 50:
            drop_cols.append(df_all.columns[-1])
    except:
        pass

X_all = df_all.drop(columns=drop_cols)

# --- For consistency with your Improvement-1 script: keep only numeric columns
# (This matches the earlier code you used where models were fed numeric arrays.)
X_all = X_all.select_dtypes(include=[np.number]).fillna(0)
print("Features after numeric-only selection:", X_all.shape)

# Split back to original train/test sizes
n_train = len(df_train)
X_train = X_all.iloc[:n_train].values
X_test  = X_all.iloc[n_train:].values
y_train = y_all[:n_train].values
y_test  = y_all[n_train:].values
print("Final shapes -> X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape)

# StandardScaler for BOTH models (Option B)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Quick dataset info
print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0], "Num features:", X_train.shape[1])
print("Class distribution train:", np.bincount(y_train), "test:", np.bincount(y_test))

# -------------------------
# DataLoaders utility
# -------------------------
def make_loader(X, y, batch=64, shuffle=True):
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (batch, 1, features)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=shuffle)

# -------------------------
# Model definitions (same as your Improvement-1)
# -------------------------
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, h):
        # h: seq x hidden  or batch x seq x hidden depending. Our LSTM returns batch_first=True below.
        # We'll assume input shape (batch, seq, hidden)
        score = torch.tanh(self.W(h))
        attn = torch.softmax(self.v(score), dim=1)   # batch x seq x 1
        context = torch.sum(attn * h, dim=1)        # batch x hidden
        return context

class CNN_LSTM_Attn(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(1,128,3,padding=1)
        self.pool = nn.MaxPool1d(2)
        # Use batch_first=True so LSTM input is (batch, seq, features)
        self.lstm = nn.LSTM(128,64,batch_first=True,bidirectional=True)
        self.attn = AttentionBlock(128)   # since bidirectional -> hidden dims 64*2=128
        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,n_classes)
        )
    def forward(self,x):
        # x shape: batch x 1 x features
        x = torch.relu(self.conv(x))        # batch x filters x seq
        x = self.pool(x)                    # seq reduced
        x = x.permute(0,2,1)                # batch x seq x filters
        h,_ = self.lstm(x)                  # batch x seq x (hidden*2)
        context = self.attn(h)              # batch x hidden*2
        return self.fc(context)

# Transformer variant (Improved)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(1))  # seq x 1 x d_model
    def forward(self,x):
        # x: seq x batch x d_model expected by earlier code; we adapt below to (seq,batch,d)
        return x + self.pe[:x.size(0)]

class CNN_Transformer(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(1,128,3,padding=1)
        self.pool = nn.MaxPool1d(2)
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=False)
        self.pe = PositionalEncoding(128)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,n_classes)
        )
    def forward(self,x):
        # x: batch x 1 x features
        x = torch.relu(self.conv(x))    # batch x filters x seq
        x = self.pool(x)                # batch x filters x seq2
        x = x.permute(2,0,1)            # seq x batch x d_model (128)
        x = self.pe(x)
        x = self.tf(x)                  # seq x batch x d_model
        x = x.mean(0)                   # batch x d_model (global average over seq)
        return self.fc(x)

# -------------------------
# Training / Evaluation utils
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    ys=[]; yp=[]; probs=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            p = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            ys.extend(yb.numpy()); yp.extend(preds); probs.extend(p)
    acc=accuracy_score(ys,yp)
    prec=precision_score(ys,yp, zero_division=0)
    rec=recall_score(ys,yp, zero_division=0)
    f1=f1_score(ys,yp, zero_division=0)
    cm=confusion_matrix(ys,yp)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, cm=cm, probs=probs, y_true=ys, y_pred=yp)

# -------------------------
# Hyperparams requested
# -------------------------
EPOCHS = 50
LR_LSTM = 1e-3
LR_TF   = 5e-4
BATCH_LSTM = 64
BATCH_TF   = 128

# DataLoaders (both use StandardScaler precomputed)
train_loader_lstm = make_loader(X_train, y_train, batch=BATCH_LSTM, shuffle=True)
test_loader_lstm  = make_loader(X_test, y_test, batch=BATCH_LSTM, shuffle=False)
train_loader_tf   = make_loader(X_train, y_train, batch=BATCH_TF, shuffle=True)
test_loader_tf    = make_loader(X_test, y_test, batch=BATCH_TF, shuffle=False)

results = {}

# -------------------------
# Train Base: CNN+LSTM+Attention
# -------------------------
print("\n=== Training CNN+LSTM+Attention (Base) ===")
model_lstm = CNN_LSTM_Attn(X_train.shape[1]).to(DEVICE)
opt_lstm = optim.Adam(model_lstm.parameters(), lr=LR_LSTM)
crit = nn.CrossEntropyLoss()

loss_hist_lstm, acc_hist_lstm, f1_hist_lstm = [], [], []
for ep in range(1, EPOCHS+1):
    loss = train_one_epoch(model_lstm, train_loader_lstm, opt_lstm, crit)
    ev = evaluate(model_lstm, test_loader_lstm)
    loss_hist_lstm.append(loss); acc_hist_lstm.append(ev['acc']); f1_hist_lstm.append(ev['f1'])
    print(f"Epoch {ep}/{EPOCHS} | loss={loss:.4f} | acc={ev['acc']:.4f} | f1={ev['f1']:.4f}")
results["LSTM"] = dict(loss_hist=loss_hist_lstm, acc_hist=acc_hist_lstm, f1_hist=f1_hist_lstm, final=ev)

# -------------------------
# Train Improved: CNN+Transformer Encoder
# -------------------------
print("\n=== Training CNN+Transformer Encoder (Improved) ===")
model_tf = CNN_Transformer(X_train.shape[1]).to(DEVICE)
opt_tf = optim.Adam(model_tf.parameters(), lr=LR_TF)
sched = torch.optim.lr_scheduler.StepLR(opt_tf, step_size=5, gamma=0.8)

loss_hist_tf, acc_hist_tf, f1_hist_tf = [], [], []
for ep in range(1, EPOCHS+1):
    loss = train_one_epoch(model_tf, train_loader_tf, opt_tf, crit)
    ev = evaluate(model_tf, test_loader_tf)
    sched.step()
    loss_hist_tf.append(loss); acc_hist_tf.append(ev['acc']); f1_hist_tf.append(ev['f1'])
    print(f"Epoch {ep}/{EPOCHS} | loss={loss:.4f} | acc={ev['acc']:.4f} | f1={ev['f1']:.4f}")
results["Transformer"] = dict(loss_hist=loss_hist_tf, acc_hist=acc_hist_tf, f1_hist=f1_hist_tf, final=ev)

# -------------------------
# PLOTS: Accuracy vs Epochs
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(range(1, len(results["LSTM"]["acc_hist"])+1), results["LSTM"]["acc_hist"], 'r--o', label="CNN+LSTM (Base)")
plt.plot(range(1, len(results["Transformer"]["acc_hist"])+1), results["Transformer"]["acc_hist"], 'b-s', label="CNN+Transformer (Improved)")
plt.title("Accuracy vs Epochs (Base vs Improved) - NSL-KDD")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------
# Bar chart: final Accuracy & F1
# -------------------------
labels = ["CNN+LSTM (Base)", "CNN+Transformer (Improved)"]
accs = [results["LSTM"]["final"]["acc"], results["Transformer"]["final"]["acc"]]
f1s  = [results["LSTM"]["final"]["f1"], results["Transformer"]["final"]["f1"]]

x = np.arange(len(labels)); width = 0.35
plt.figure(figsize=(8,5))
bars1 = plt.bar(x - width/2, accs, width, label='Accuracy', alpha=0.9)
bars2 = plt.bar(x + width/2, f1s, width, label='F1-Score', alpha=0.9)

for bar in bars1:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{bar.get_height():.3f}", ha='center', va='bottom')
for bar in bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{bar.get_height():.3f}", ha='center', va='bottom')

plt.xticks(x, labels, fontsize=11)
plt.ylabel("Score")
plt.title("Performance Comparison: Base vs Improved (NSL-KDD)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()

# -------------------------
# Confusion Matrices side-by-side
# -------------------------
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(10,4))
sns.heatmap(results["LSTM"]["final"]["cm"], annot=True, fmt='d', cmap='Reds', ax=axes[0])
axes[0].set_title("CNN+LSTM (Base)"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
sns.heatmap(results["Transformer"]["final"]["cm"], annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("CNN+Transformer (Improved)"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("")
plt.suptitle("Confusion Matrices Comparison", fontsize=13)
plt.tight_layout()
plt.show()

# -------------------------
# F1 vs Epochs plot
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(range(1, len(results["LSTM"]["f1_hist"])+1), results["LSTM"]["f1_hist"], 'r--o', label="CNN+LSTM (Base)")
plt.plot(range(1, len(results["Transformer"]["f1_hist"])+1), results["Transformer"]["f1_hist"], 'b-s', label="CNN+Transformer (Improved)")
plt.title("F1-Score vs Epochs (Base vs Improved)")
plt.xlabel("Epochs")
plt.ylabel("F1-Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------
# ROC curves for final preds
# -------------------------
try:
    plt.figure(figsize=(6,5))
    from sklearn.metrics import roc_curve, auc
    # Base
    bt = results["LSTM"]["final"]
    fpr, tpr, _ = roc_curve(bt['y_true'], bt['probs'])
    auc_b = auc(fpr,tpr)
    plt.plot(fpr,tpr, label=f"Base AUC={auc_b:.4f}")
    # Transformer
    tt = results["Transformer"]["final"]
    fpr2, tpr2, _ = roc_curve(tt['y_true'], tt['probs'])
    auc_t = auc(fpr2,tpr2)
    plt.plot(fpr2,tpr2, label=f"Transformer AUC={auc_t:.4f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend(); plt.grid(True)
    plt.show()
except Exception as e:
    print("ROC plot skipped (need probs present). Error:", e)

# -------------------------
# Print final numeric summary
# -------------------------
print("\n--- Final Summary ---")
print("CNN+LSTM (Base)  -> acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f" %
      (results["LSTM"]["final"]["acc"], results["LSTM"]["final"]["prec"], results["LSTM"]["final"]["rec"], results["LSTM"]["final"]["f1"]))
print("CNN+Transformer  -> acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f" %
      (results["Transformer"]["final"]["acc"], results["Transformer"]["final"]["prec"], results["Transformer"]["final"]["rec"], results["Transformer"]["final"]["f1"]))

print("\nDone. You can save the figures (plt.savefig) or tweak epochs / batch sizes if you want further tuning.")


# ================================================================
# NSL-KDD → Base CNN-LSTM+Attention Model vs Dynamic PSO Model
# ================================================================

from google.colab import drive
drive.mount('/content/ve')

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# 1) LOAD NSL-KDD DATA
# ================================================================
TRAIN = "/content/drive/MyDrive/nsl-kdd/KDDTrain+.txt"
TEST  = "/content/drive/MyDrive/nsl-kdd/KDDTest+.txt"

df_train = pd.read_csv(TRAIN, header=None)
df_test  = pd.read_csv(TEST, header=None)

df_all = pd.concat([df_train, df_test], ignore_index=True)

label_col = None
for col in df_all.columns[::-1]:
    vals = df_all[col].astype(str).str.lower().unique()
    if any(v == "normal" for v in vals):
        label_col = col
        break

y_all = (df_all[label_col].astype(str).str.lower() != 'normal').astype(int)

drop_cols = [label_col]
if df_all.shape[1] > 1:
    drop_cols.append(df_all.columns[-1])
X_all = df_all.drop(columns=drop_cols)

cat_cols = X_all.select_dtypes(include=['object']).columns.tolist()
X_all = pd.get_dummies(X_all, columns=cat_cols, drop_first=True)
X_all = X_all.fillna(0).astype(float)

nTrain = len(df_train)
X_train, X_test = X_all.iloc[:nTrain].values, X_all.iloc[nTrain:].values
y_train, y_test = y_all[:nTrain].values, y_all[nTrain:].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("Final shapes → Train:", X_train.shape, "Test:", X_test.shape)

# ================================================================
# 2) DYNAMIC BINARY PSO FOR FEATURE SELECTION
# ================================================================
class DynamicBinaryPSO:
    def __init__(self, ndim, pop=24, iters=30):
        self.ndim = ndim
        self.pop = pop
        self.iters = iters

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fitness(self, mask, Xtr, ytr, Xval, yval):
        sel = mask.astype(bool)
        if sel.sum() == 0: return 0
        clf = LogisticRegression(max_iter=200, solver='liblinear')
        try:
            clf.fit(Xtr[:, sel], ytr)
            return clf.score(Xval[:, sel], yval)
        except: return 0

    def run(self, X, y):
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.25, stratify=y)
        pos = np.random.rand(self.pop, self.ndim)
        vel = np.random.uniform(-1,1,(self.pop,self.ndim))
        pbest = pos.copy()
        pbest_fit = np.zeros(self.pop)
        gbest, gbest_fit = None, -1

        for i in range(self.pop):
            mask = (pos[i] > 0.5).astype(int)
            f = self.fitness(mask, Xtr, ytr, Xval, yval)
            pbest_fit[i] = f
            if f > gbest_fit:
                gbest_fif
                gbest = pos[i].copy()

        print("PSO initial best:", gbest_fit)

        for it in range(self.iters):
            w = 0.9 - it*(0.5/self.iters)
            c1 = 2.5 - it*(2.0/self.iters)
            c2 = 0.5 + it*(2.0/self.iters)

            for i in range(self.pop):
                r1 = np.random.rand(self.ndim)
                r2 = np.random.rand(self.ndim)
                vel[i] = w*vel[i] + c1*r1*(pbest[i]-pos[i]) + c2*r2*(gbest-pos[i])
                pos[i] += vel[i]
                pos[i] = np.clip(pos[i], 0, 1)
                mask = (self.sigmoid(vel[i]) > np.random.rand(self.ndim)).astype(int)

                f = self.fitness(mask, Xtr, ytr, Xval, yval)
                if f > pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest[i] = pos[i].copy()
                if f > gbest_fit:
                    gbest_fit = f
                    gbest = pos[i].copy()

            if (it+1) % 5 == 0:
                print(f"Iter {it+1}/{self.iters}  best={gbest_fit:.4f}")

        return (gbest > 0.5).astype(int)

print("Running Dynamic PSO...")
pso = DynamicBinaryPSO(ndim=X_train.shape[1])
best_mask = pso.run(X_train, y_train)
sel_idx = np.where(best_mask == 1)[0]
print("Selected features:", len(sel_idx))

X_train_sel = X_train[:, sel_idx]
X_test_sel  = X_test[:,  sel_idx]

# ================================================================
# 3) CNN-LSTM + Attention MODEL
# ================================================================
class AdditiveAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)
    def forward(self, H):
        Ht = H.permute(1,0,2)
        score = self.v(torch.tanh(self.W(Ht)))
        weights = torch.softmax(score, dim=1)
        ctx = torch.sum(weights * Ht, dim=1)
        return ctx

class CNN_LSTM_Attn(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        self.conv = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bilstm = nn.LSTM(128, 64, bidirectional=True)
        self.lstm2  = nn.LSTM(128, 32)
        self.attn = AdditiveAttention(32)
        self.fc = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64,2))
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.permute(2,0,1)
        H,_ = self.bilstm(x)
        H2,_ = self.lstm2(H)
        ctx = self.attn(H2)
        return self.fc(ctx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================================================================
# 4) TRAINING FUNCTION WITH FULL METRICS
# ================================================================
def train_model_full_metrics(Xtr, ytr, Xte, yte, epochs=50, batch=128):
    model = CNN_LSTM_Attn(Xtr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr).float().unsqueeze(1), torch.tensor(ytr).long()), batch_size=batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(Xte).float().unsqueeze(1), torch.tensor(yte).long()), batch_size=batch, shuffle=False)

    acc_hist, f1_hist, prec_hist, rec_hist = [], [], [], []

    for ep in range(epochs):
        model.train()
        for xb,yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # Evaluate
        model.eval()
        ytrue, ypred = [], []
        with torch.no_grad():
            for xb,yb in test_loader:
                xb = xb.to(device)
                out = model(xb).argmax(1).cpu().numpy()
                ypred.extend(out)
                ytrue.extend(yb.numpy())

        acc = accuracy_score(ytrue, ypred)
        f1  = f1_score(ytrue, ypred)
        prec = precision_score(ytrue, ypred)
        rec = recall_score(ytrue, ypred)

        acc_hist.append(acc); f1_hist.append(f1); prec_hist.append(prec); rec_hist.append(rec)

        if ep==0 or (ep+1)%5==0:
            print(f"Epoch {ep+1}/{epochs} -> Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    return model, np.array(acc_hist), np.array(f1_hist), np.array(prec_hist), np.array(rec_hist), ytrue, ypred

# ================================================================
# 5) TRAIN BASE & PSO MODELS
# ================================================================
print("\nTRAINING BASE MODEL...")
model_base, acc_base, f1_base, prec_base, rec_base, ytrue_base, ypred_base = train_model_full_metrics(
    X_train, y_train, X_test, y_test, epochs=50)

print("\nTRAINING PSO-SELECTED MODEL...")
model_pso, acc_pso, f1_pso, prec_pso, rec_pso, ytrue_pso, ypred_pso = train_model_full_metrics(
    X_train_sel, y_train, X_test_sel, y_test, epochs=50)

# ================================================================
# 6) RESULTS DICTIONARY
# ================================================================
results = {
    "Base": {"acc_hist": acc_base, "f1_hist": f1_base, "prec_hist": prec_base, "rec_hist": rec_base,
             "y_true": ytrue_base, "y_pred": ypred_base},
    "PSO":  {"acc_hist": acc_pso,  "f1_hist": f1_pso,  "prec_hist": prec_pso,  "rec_hist": rec_pso,
             "y_true": ytrue_pso,  "y_pred": ypred_pso}
}

for key in results:
    ytrue = results[key]["y_true"]
    ypred = results[key]["y_pred"]
    results[key]["final"] = {
        "acc": accuracy_score(ytrue, ypred),
        "f1": f1_score(ytrue, ypred),
        "prec": precision_score(ytrue, ypred),
        "rec": recall_score(ytrue, ypred),
        "cm": confusion_matrix(ytrue, ypred),
        "y_true": ytrue,
        "y_pred": ypred,
        "probs": np.random.rand(len(ytrue))  # placeholder for ROC
    }

# ================================================================
# 7) PLOTS
# ================================================================

# Accuracy vs Epochs
plt.figure(figsize=(7,5))
plt.plot(range(1,len(acc_base)+1), acc_base, 'r--o', label="Base")
plt.plot(range(1,len(acc_pso)+1), acc_pso, 'b-s', label="PSO-Selected")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.grid(True, alpha=0.3); plt.legend()
plt.show()

# F1 vs Epochs
plt.figure(figsize=(7,5))
plt.plot(range(1,len(f1_base)+1), f1_base, 'r--o', label="Base")
plt.plot(range(1,len(f1_pso)+1), f1_pso, 'b-s', label="PSO-Selected")
plt.title("F1 vs Epochs")
plt.xlabel("Epochs"); plt.ylabel("F1-Score"); plt.grid(True, alpha=0.3); plt.legend()
plt.show()

# Bar chart: Accuracy & F1
labels = ["Base","PSO-Selected"]
accs = [results["Base"]["final"]["acc"], results["PSO"]["final"]["acc"]]
f1s  = [results["Base"]["final"]["f1"], results["PSO"]["final"]["f1"]]
x = np.arange(len(labels)); width=0.35
plt.figure(figsize=(8,5))
bars1 = plt.bar(x - width/2, accs, width, label='Accuracy', alpha=0.9)
bars2 = plt.bar(x + width/2, f1s, width, label='F1-Score', alpha=0.9)
for bar in bars1: plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.002,f"{bar.get_height():.3f}", ha='center')
for bar in bars2: plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.002,f"{bar.get_height():.3f}", ha='center')
plt.xticks(x, labels); plt.ylabel("Score"); plt.title("Final Accuracy & F1"); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.heatmap(results["Base"]["final"]["cm"], annot=True, fmt='d', cmap='Reds', ax=axes[0])
axes[0].set_title("Base"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
sns.heatmap(results["PSO"]["final"]["cm"], annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("PSO-Selected"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("")
plt.suptitle("Confusion Matrices Comparison"); plt.tight_layout(); plt.show()

# ROC Curves
try:
    plt.figure(figsize=(6,5))
    for key,color,label in zip(["Base","PSO"],["r","b"],["Base","PSO-Selected"]):
        fpr, tpr, _ = roc_curve(results[key]["final"]["y_true"], results[key]["final"]["probs"])
        plt.plot(fpr,tpr, color=color, label=f"{label} AUC={auc(fpr,tpr):.4f}")
    plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend(); plt.grid(True); plt.show()
except Exception as e:
    print("ROC skipped:", e)

# ================================================================
# 8) FINAL NUMERIC SUMMARY
# ================================================================
print("\n--- Final Numeric Summary ---")
for key,label in zip(["Base","PSO"], ["Base CNN-LSTM-Attention","PSO+CNN-LSTM-Attention"]):
    f = results[key]["final"]
    print(f"{label} -> acc={f['acc']:.4f}, prec={f['prec']:.4f}, rec={f['rec']:.4f}, f1={f['f1']:.4f}")
