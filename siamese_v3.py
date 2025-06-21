import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import random
from sklearn.metrics import roc_auc_score


# ==== 1. Load dữ liệu ====
with open("qr_codes_29.pickle", "rb") as f:
    qr_codes = pickle.load(f)
with open("qr_codes_29_labels.pickle", "rb") as f:
    labels = pickle.load(f)

X = qr_codes.astype(np.float32)
y = np.array(labels)
X = X.reshape(-1, 1, 69, 69)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 2. Tạo dataset dạng cặp ====
class SiameseDataset(Dataset):
    def __init__(self, X, y, n_pairs=20000):
        self.X = torch.tensor(X)
        self.y = y
        self.pairs = []
        self.labels = []

        class0 = np.where(y == 0)[0]
        class1 = np.where(y == 1)[0]

        for _ in range(n_pairs // 2):
            # Positive pair
            cls = random.choice([0, 1])
            idx = np.random.choice(np.where(y == cls)[0], 2, replace=False)
            self.pairs.append((self.X[idx[0]], self.X[idx[1]]))
            self.labels.append(0)

            # Negative pair
            i0 = np.random.choice(class0)
            i1 = np.random.choice(class1)
            self.pairs.append((self.X[i0], self.X[i1]))
            self.labels.append(1)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        return x1, x2, torch.tensor(self.labels[idx]).float()
    def __len__(self):
        return len(self.labels)

# ==== 3. CNN encoder dùng chung ====
class QR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128)
        )

    def forward(self, x):
        return self.encoder(x)

# ==== 4. Contrastive Loss ====
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = torch.norm(out1 - out2, dim=1)
        loss = (1 - label) * dist.pow(2) + label * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()

# ==== 5. Training Siamese ====
dataset = SiameseDataset(X, y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = QR_Encoder().to(device)
loss_fn = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    total_loss = 0
    for x1, x2, label in loader:
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        out1 = model(x1)
        out2 = model(x2)
        loss = loss_fn(out1, out2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")


# ==== 6. Đánh giá trên cặp test ====
print("\n📊 Đánh giá ROC-AUC trên cặp QR test...")

test_dataset = SiameseDataset(X, y, n_pairs=10000)
test_loader = DataLoader(test_dataset, batch_size=128)

model.eval()
all_dist = []
all_labels = []

with torch.no_grad():
    for x1, x2, label in test_loader:
        x1, x2 = x1.to(device), x2.to(device)
        out1 = model(x1)
        out2 = model(x2)
        dist = torch.norm(out1 - out2, dim=1).cpu().numpy()
        all_dist.extend(dist)
        all_labels.extend(label.numpy())

auc = roc_auc_score(all_labels, all_dist)
print(f"🎯 Siamese CNN AUC (distance-based): {auc:.4f}")



# ==== 7. Metrics, ROC & t-SNE =================================================
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve)
from sklearn.manifold import TSNE
import os

# -- 7.1  Chọn ngưỡng tốt nhất theo F1 ----------------------------------------
fpr, tpr, thresholds = roc_curve(all_labels, all_dist, pos_label=1)
f1s = []
for thr in thresholds:
    preds = (all_dist >= thr).astype(int)   # >= thr => dự đoán "khác nhau" (label 1)
    f1s.append(f1_score(all_labels, preds))
best_idx = np.argmax(f1s)
best_thr = thresholds[best_idx]

preds = (all_dist >= best_thr).astype(int)
acc  = accuracy_score(all_labels, preds)
prec = precision_score(all_labels, preds)
rec  = recall_score(all_labels, preds)
f1   = f1_score(all_labels, preds)

print(f"\n⚙️  Threshold (max F1): {best_thr:.4f}")
print(f"📈 Accuracy : {acc:.4f}")
print(f"🎯 Precision: {prec:.4f}")
print(f"🔄 Recall   : {rec:.4f}")
print(f"🏅 F1-score : {f1:.4f}")

# -- 7.2  Lưu ROC curve --------------------------------------------------------
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], "--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve – Siamese QR")
plt.legend()
plt.tight_layout()
plt.savefig("roc_qr.png", dpi=300)
plt.close()
print("💾 ROC curve saved to roc_qr.png")

# -- 7.3  t-SNE của toàn bộ embedding -----------------------------------------
with torch.no_grad():
    full_embeddings = model(torch.tensor(X).to(device)).cpu().numpy()

tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=42)
embed_2d = tsne.fit_transform(full_embeddings)

plt.figure(figsize=(6,6))
plt.scatter(embed_2d[y==0,0], embed_2d[y==0,1], s=6, alpha=0.7, label="Class 0")
plt.scatter(embed_2d[y==1,0], embed_2d[y==1,1], s=6, alpha=0.7, label="Class 1")
plt.legend()
plt.title("t-SNE of QR embeddings")
plt.tight_layout()
plt.savefig("tsne_qr.png", dpi=300)
plt.close()
print("💾 t-SNE plot saved to tsne_qr.png")

