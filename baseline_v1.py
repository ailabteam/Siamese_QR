# qr_baseline_models.py

import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torchvision.models import mobilenet_v2, resnet18, vit_b_16

# ==== 1. Load dữ liệu ====
with open("qr_codes_29.pickle", "rb") as f:
    qr_codes = pickle.load(f)
with open("qr_codes_29_labels.pickle", "rb") as f:
    labels = pickle.load(f)

X = qr_codes.astype(np.float32)
y = np.array(labels)
X = X.reshape(-1, 69, 69)  # ảnh đơn kênh

# ==== 2. Dataset ====
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor()
])

class QRDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.X)

# ==== 3. Models ====
class CNN_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x))

class MobileNetQR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ResNetQR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ViTQR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(pretrained=True)
        self.model.conv_proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.model.heads = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ==== 4. Huấn luyện và đánh giá ====
def train_and_eval(model_class, X, y, is_rgb=False):
    print(f"\n🧠 Training model: {model_class.__name__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if is_rgb:
        train_ds = QRDataset(X_train, y_train, transform=transform)
        test_ds = QRDataset(X_test, y_test, transform=transform)
    else:
        train_ds = QRDataset(X_train, y_train, transform=T.Compose([T.ToTensor()]))
        test_ds = QRDataset(X_test, y_test, transform=T.Compose([T.ToTensor()]))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = model_class().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(30):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            out = model(xb).squeeze(1)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).squeeze(1).cpu().numpy()
            preds.extend(out)
            true.extend(yb.numpy())

    auc = roc_auc_score(true, preds)
    print(f"🎯 {model_class.__name__} AUC: {auc:.4f}")

# ==== 5. Chạy tất cả baseline ====
print("\n--- Running Baseline Models ---\n")
train_and_eval(CNN_Baseline, X, y, is_rgb=False)
train_and_eval(MobileNetQR, X, y, is_rgb=True)
train_and_eval(ResNetQR, X, y, is_rgb=True)
train_and_eval(ViTQR, X, y, is_rgb=True)

