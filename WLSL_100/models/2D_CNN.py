import torch
from torch.utils.data import Dataset
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

class SignDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = torch.load(row.tensor_path)
        x = x.unsqueeze(1)
        y = row.class_id
        return x, y

"data/annotations/processed_wlsl_100.csv"


### Baseline model without batch norm nothing else
class FrameCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)

## Actual base line model
class TemporalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.frame_cnn = FrameCNN()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        feats = self.frame_cnn(x)        

        feats = feats.view(B, T, -1)
        feats = feats.mean(dim=1)        

        out = self.classifier(feats)
        return out


### training model
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


