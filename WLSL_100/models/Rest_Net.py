import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import models


class SignDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = torch.load(row.tensor_path)     # (32,112,112)
        x = x.unsqueeze(1)                  # (32,1,112,112)
        x = x.repeat(1, 3, 1, 1)            # (32,3,112,112)

        y = int(row.class_id)
        return x, y


class ResNetBackbone(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove final classifier
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # (512,1,1)

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B*T, 3, 112, 112)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B*T, 512)
        return x


class ResNet_TemporalConv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = ResNetBackbone(freeze=True)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, T, 3, 112, 112)
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        feats = self.backbone(x)          # (B*T, 512)

        feats = feats.view(B, T, 512)
        feats = feats.transpose(1, 2)     # (B, 512, T)

        feats = self.temporal_conv(feats) # (B, 256, T)
        feats = feats.mean(dim=2)         # temporal pooling

        feats = self.dropout(feats)
        out = self.classifier(feats)
        return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total



def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.4f}"
        )


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

train_ds = SignDataset("/home/vedant/Code/AI_ML_RL_RO/Projects/sign_language_detection/WLSL_100/train_final.csv")
val_ds   = SignDataset("/home/vedant/Code/AI_ML_RL_RO/Projects/sign_language_detection/WLSL_100/val_final.csv")

num_classes = train_ds.df.class_id.nunique()
print("Number of classes:", num_classes)

# Sanity checks
assert train_ds.df.class_id.min() == 0
assert train_ds.df.class_id.max() == num_classes - 1
assert set(train_ds.df.class_id.unique()) == set(val_ds.df.class_id.unique())

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

model = ResNet_TemporalConv(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=20
)

print("Training complete")


