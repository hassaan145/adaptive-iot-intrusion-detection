
# -------------------- Mount Drive (uncomment if needed) --------------------
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# -------------------- Imports --------------------
import os, time, random, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -------------------- Settings --------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -------------------- Paths --------------------
TRAIN_PATH = "/content/drive/MyDrive/nsl-kdd/KDDTrain+.txt"
TEST_PATH  = "/content/drive/MyDrive/nsl-kdd/KDDTest+.txt"

# -------------------- Load with column names (robust) --------------------
# NSL-KDD many variants; we'll load without header then detect label column
df_train = pd.read_csv(TRAIN_PATH, header=None)
df_test  = pd.read_csv(TEST_PATH, header=None)
print("Raw shapes -> train:", df_train.shape, "test:", df_test.shape)

# Combine to preprocess consistently
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
print("Combined rows:", df_all.shape)

# Heuristic: find label column by searching for 'normal' string in columns
label_col = None
for col in df_all.columns[::-1]:
    vals = df_all[col].astype(str).str.lower().str.strip().unique()
    # test common variants
    if any('normal' in v for v in vals):
        label_col = col
        break
if label_col is None:
    # fallback: second last
    label_col = df_all.columns[-1]
print("Detected label column index:", label_col)

# Clean label strings: strip spaces and trailing dots
df_all[label_col] = df_all[label_col].astype(str).str.strip().str.lower().str.rstrip('.')

# Now build binary label: normal -> 0, others -> 1
y_all = (df_all[label_col] != 'normal').astype(int).values

# Drop the label column from features
X_all = df_all.drop(columns=[label_col]).copy()

# If last column looks like difficulty (small set of ints), drop it
if X_all.shape[1] >= 1:
    last_vals = X_all.iloc[:, -1].astype(str)
    try:
        num_uniq = len(pd.Series(last_vals).unique())
        if num_uniq < 50:
            X_all = X_all.iloc[:, :-1]
    except Exception:
        pass

# Convert object columns to dummies
obj_cols = X_all.select_dtypes(include=['object','category']).columns.tolist()
if obj_cols:
    X_all = pd.get_dummies(X_all, columns=obj_cols, drop_first=True)

# Keep numeric and fill NA
X_all = X_all.select_dtypes(include=[np.number]).fillna(0)
print("Features after preprocessing:", X_all.shape)

# Split back into train/test
n_train = len(df_train)
X_train = X_all.iloc[:n_train].values
X_test  = X_all.iloc[n_train:].values
y_train = y_all[:n_train]
y_test  = y_all[n_train:]
print("Final shapes -> Train:", X_train.shape, "Test:", X_test.shape)
print("Train class counts:", np.bincount(y_train), "Test class counts:", np.bincount(y_test))

# If classes are single-valued, raise an informative error
if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
    raise RuntimeError("Label mapping produced single class. Inspect raw label values and mapping logic.")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------- Fast Dynamic Binary PSO --------------------
class FastDynamicPSO:
    def __init__(self, n_dim, pop=12, iters=12, w0=0.9, w1=0.4, c1s=2.5, c1e=0.5, c2s=0.5, c2e=2.5, seed=SEED):
        self.n = n_dim
        self.pop = pop
        self.iters = iters
        self.w0, self.w1 = w0, w1
        self.c1s, self.c1e = c1s, c1e
        self.c2s, self.c2e = c2s, c2e
        random.seed(seed); np.random.seed(seed)
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def fitness(self, mask, Xtr, ytr, Xval, yval):
        sel = mask.astype(bool)
        if sel.sum() == 0:
            return 0.0
        clf = LogisticRegression(max_iter=200, solver='liblinear')
        try:
            clf.fit(Xtr[:, sel], ytr)
            return clf.score(Xval[:, sel], yval)
        except Exception:
            return 0.0
    def run(self, X, y, val_frac=0.25, verbose=True):
        Xtr_i, Xval_i, ytr_i, yval_i = train_test_split(X, y, test_size=val_frac, random_state=SEED, stratify=y)
        n = self.n
        pos = np.random.rand(self.pop, n)
        vel = np.random.uniform(-1,1,size=(self.pop,n))
        pbest_pos = pos.copy()
        pbest_fit = np.zeros(self.pop)
        gbest_pos = None
        gbest_fit = -1.0
        # initial eval
        for i in range(self.pop):
            mask = (pos[i] > 0.5).astype(int)
            pbest_fit[i] = self.fitness(mask, Xtr_i, ytr_i, Xval_i, yval_i)
            if pbest_fit[i] > gbest_fit:
                gbest_fit = pbest_fit[i]; gbest_pos = pos[i].copy()
        if verbose: print("PSO init best:", gbest_fit, "sel:", (gbest_pos>0.5).sum())
        for it in range(self.iters):
            t = it / max(1, self.iters-1)
            w = self.w0*(1-t) + self.w1*t
            c1 = self.c1s*(1-t) + self.c1e*t
            c2 = self.c2s*(1-t) + self.c2e*t
            for i in range(self.pop):
                r1 = np.random.rand(n); r2 = np.random.rand(n)
                vel[i] = w*vel[i] + c1*r1*(pbest_pos[i]-pos[i]) + c2*r2*(gbest_pos-pos[i])
                pos[i] = np.clip(pos[i] + vel[i], 0.0, 1.0)
                mask = (self._sigmoid(vel[i]) > np.random.rand(n)).astype(int)
                f = self.fitness(mask, Xtr_i, ytr_i, Xval_i, yval_i)
                if f > pbest_fit[i]:
                    pbest_fit[i] = f; pbest_pos[i] = pos[i].copy()
                if f > gbest_fit:
                    gbest_fit = f; gbest_pos = pos[i].copy()
            if verbose and ((it+1) % 3 == 0 or it==0 or it==self.iters-1):
                print(f"Iter {it+1}/{self.iters} best={gbest_fit:.4f} sel={(gbest_pos>0.5).sum()}")
        best_mask = (gbest_pos > 0.5).astype(int)
        return best_mask, gbest_fit

print("\nRunning fast Dynamic PSO (pop=12, iters=12)...")
pso = FastDynamicPSO(n_dim=X_train.shape[1], pop=12, iters=12)
t0 = time.time()
best_mask, best_val = pso.run(X_train, y_train, val_frac=0.25, verbose=True)
print("PSO done in %.1f sec. best_val=%.4f selected=%d" % (time.time()-t0, best_val, best_mask.sum()))
sel_idx = np.where(best_mask==1)[0]
if sel_idx.size == 0:
    raise RuntimeError("PSO selected zero features - increase iters/pop or change logic.")
print("Selected indices (count):", sel_idx.size)

# -------------------- Model definitions --------------------
# Simple helper to create loaders
def make_loader(X, y, batch=128, shuffle=True):
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # B x 1 x features
    yt = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

# CNN only
class SimpleCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(nn.Linear((n_features//2)*64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128,2))
    def forward(self, x):
        x = self.conv(x)          # B x 64 x (F/2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# LSTM only (treat feature vector as sequence)
class SimpleLSTM(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64,2))
    def forward(self, x):
        # x: B x 1 x F -> B x F x 1
        x = x.permute(0,2,1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# CNN-LSTM base (paper style)
class CNN_LSTM_Base(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64,2))
    def forward(self, x):
        x = self.cnn(x)           # B x 128 x (F/2)
        x = x.permute(0,2,1)      # B x Seq x 128
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# Attention module
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, H):
        Hb = H.permute(1,0,2)  # batch x seq x hidden
        scores = self.v(torch.tanh(self.W(Hb)))  # batch x seq x1
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * Hb, dim=1)
        return context, weights

class CNN_LSTM_Attn(nn.Module):
    def __init__(self, n_features, cnn_filters=128, lstm_hidden=64, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, cnn_filters, kernel_size=3, padding=1)

        # REMOVE POOLING to avoid mismatched sequence length
        # self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True
        )

        self.attn = AdditiveAttention(lstm_hidden * 2)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))      # batch x filters x seq
        # x = self.pool(x)                 # REMOVED
        x = x.permute(2, 0, 1)
        H, _ = self.lstm(x)
        context, _ = self.attn(H)
        out = self.fc(context)
        return out

# Transformer encoder (encoder only)
class TransformerEncoderOnly(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)  # map each feature scalar to d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64,2))
    def forward(self, x):
        # x: B x 1 x F -> B x F x 1
        x = x.permute(0,2,1)
        x = self.input_proj(x)   # B x F x d_model
        H = self.encoder(x)      # B x F x d_model
        # simple pooling over sequence
        Hp = H.mean(dim=1)       # B x d_model
        return self.fc(Hp)

# -------------------- Train & Eval utility --------------------
def train_model_torch(model, Xtr, ytr, Xte, yte, n_epochs=8, batch_size=128, lr=1e-3):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_loader = make_loader(Xtr, ytr, batch=batch_size, shuffle=True)
    test_loader = make_loader(Xte, yte, batch=batch_size, shuffle=False)
    loss_hist=[]; acc_hist=[]; f1_hist=[]
    for ep in range(1, n_epochs+1):
        model.train()
        running=0.0
        for xb,yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        avg_loss = running / len(train_loader.dataset)
        # eval
        model.eval()
        y_true=[]; y_pred=[]
        with torch.no_grad():
            for xb,yb in test_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                preds = out.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds.tolist()); y_true.extend(yb.numpy().tolist())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        loss_hist.append(avg_loss); acc_hist.append(acc); f1_hist.append(f1)
        print(f"Epoch {ep}/{n_epochs}  Accuracy={acc:.4f}  F1={f1:.4f}")
    # final metrics + probs
    model.eval()
    y_true_all=[]; y_pred_all=[]; probs=[]
    with torch.no_grad():
        for xb,yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            p = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            probs.extend(p.tolist()); y_pred_all.extend(preds.tolist()); y_true_all.extend(yb.numpy().tolist())
    metrics = {
        'accuracy': accuracy_score(y_true_all, y_pred_all),
        'precision': precision_score(y_true_all, y_pred_all, zero_division=0),
        'recall': recall_score(y_true_all, y_pred_all, zero_division=0),
        'f1': f1_score(y_true_all, y_pred_all),
        'confusion': confusion_matrix(y_true_all, y_pred_all),
        'probs': probs,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'loss_history': loss_hist,
        'acc_history': acc_hist,
        'f1_history': f1_hist
    }
    return model, metrics

# -------------------- Run & compare models (FAST) --------------------
models_to_run = {
    "CNN": SimpleCNN(X_train.shape[1]),
    "LSTM": SimpleLSTM(X_train.shape[1]),
    "CNN-LSTM": CNN_LSTM_Base(X_train.shape[1]),
    "CNN-LSTM-Attn": CNN_LSTM_Attn(X_train.shape[1]),
    "TransformerEncoder": TransformerEncoderOnly(X_train.shape[1])
}

results = {}
EPOCHS = 8
for name, m in models_to_run.items():
    print(f"\nTRAINING {name} ...")
    mx, met = train_model_torch(m, X_train, y_train, X_test, y_test, n_epochs=EPOCHS, batch_size=128, lr=1e-3)
    results[name] = met

# Now run PSO-selected CNN-LSTM
print("\nTRAINING CNN-LSTM on PSO-selected features ...")
X_tr_sel = X_train[:, sel_idx]; X_te_sel = X_test[:, sel_idx]
model_pso, metrics_pso = train_model_torch(CNN_LSTM_Base(X_tr_sel.shape[1]), X_tr_sel, y_train, X_te_sel, y_test, n_epochs=EPOCHS, batch_size=128, lr=1e-3)
results["PSO_CNN-LSTM"] = metrics_pso

# -------------------- Summary table --------------------
import pandas as pd
rows=[]
for k,v in results.items():
    rows.append([k, v['accuracy'], v['precision'], v['recall'], v['f1']])
df_res = pd.DataFrame(rows, columns=["Model","Accuracy","Precision","Recall","F1"])
print("\n=== Comparison Table ===")
print(df_res)

# -------------------- Plot Accuracy vs Epoch (same plane) --------------------
plt.figure(figsize=(8,5))
for k,v in results.items():
    plt.plot(v['acc_history'], marker='o', label=k)
plt.title("Accuracy vs Epoch (All models)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# -------------------- Confusion matrices (print) --------------------
for k,v in results.items():
    print(f"\nConfusion Matrix: {k}")
    print(v['confusion'])
