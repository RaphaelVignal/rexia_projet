import os
import random
from pathlib import Path

import cv2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm, trange

# ── Config ────────────────────────────────────────────────────────────────────

SEED       = 42
BATCH_SIZE = 256
EPOCHS     = 5
LR         = 1e-3
BASE_DIR   = os.getcwd()
CELEBA_DIR = os.path.join(BASE_DIR, "data", "CelebA")
CSV_FILE   = os.path.join(CELEBA_DIR, "list_attr_celeba.csv")
IMG_DIR    = os.path.join(CELEBA_DIR, "img_align_celeba/img_align_celeba")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ────────────────────────────────────────────────────────────────

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Face crop ─────────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_face(img_pil):
    img_np  = np.array(img_pil)
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces   = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_np.shape[1], x + w + margin)
        y2 = min(img_np.shape[0], y + h + margin)
        return Image.fromarray(img_np[y1:y2, x1:x2])

    return img_pil

# ── Dataset ───────────────────────────────────────────────────────────────────

class CelebADataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, crop=False):
        self.data      = pd.read_csv(csv_file)
        self.img_dir   = img_dir
        self.transform = transform
        self.crop      = crop

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, row["image_id"])).convert("RGB")
        if self.crop:
            image = crop_face(image)
        label  = torch.tensor((row["Smiling"] + 1) / 2, dtype=torch.float32)
        gender = torch.tensor((row["Male"]    + 1) / 2, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label, gender

# ── Loaders ───────────────────────────────────────────────────────────────────

def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=4, pin_memory=True)

def build_loaders(crop=False):
    random.seed(SEED)

    ds_train    = CelebADataset(CSV_FILE, IMG_DIR, transform=transform_train, crop=crop)
    ds_inferval = CelebADataset(CSV_FILE, IMG_DIR, transform=transform_infer,  crop=crop)

    total      = len(ds_train)
    indices    = list(range(total))
    random.shuffle(indices)

    train_size = int(0.8 * total)
    val_size   = int(0.1 * total)

    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]

    return (
        make_loader(Subset(ds_train,    train_idx), shuffle=True),
        make_loader(Subset(ds_inferval, val_idx),   shuffle=False),
        make_loader(Subset(ds_inferval, test_idx),  shuffle=False),
        test_idx,
    )

# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, labels, _ in pbar:
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += ((preds.sigmoid() > 0.5) == labels).sum().item()
        total      += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


def eval_model(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_genders = [], [], []
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for imgs, labels, genders in pbar:
            imgs   = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            preds  = model(imgs)
            total_loss += criterion(preds, labels).item()
            correct    += ((preds.sigmoid() > 0.5) == labels).sum().item()
            total      += labels.size(0)
            all_preds.append(preds.sigmoid().cpu())
            all_labels.append(labels.cpu())
            all_genders.append(genders.cpu())
            pbar.set_postfix(loss=f"{total_loss/len(loader):.4f}", acc=f"{correct/total:.4f}")
    return (
        total_loss / len(loader),
        correct / total,
        torch.cat(all_preds).squeeze(),
        torch.cat(all_labels).squeeze(),
        torch.cat(all_genders).squeeze(),
    )

# ── Run ───────────────────────────────────────────────────────────────────────

def run_training(save_path, crop=False):
    print(f"\n--- Entraînement {'avec' if crop else 'sans'} crop → {save_path} ---")
    print(f"Device : {device}")

    train_loader, val_loader, test_loader, test_idx = build_loaders(crop=crop)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    model    = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for e in trange(EPOCHS, desc="Epochs"):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _, _ = eval_model(model, val_loader, criterion)
        train_losses.append(train_loss); train_accs.append(train_acc)
        val_losses.append(val_loss);     val_accs.append(val_acc)
        print(f"Epoch {e+1}/{EPOCHS}  "
              f"train loss {train_loss:.4f} acc {train_acc:.4f}  "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

    test_loss, test_acc, all_preds, all_labels, all_genders = eval_model(
        model, test_loader, criterion)
    print(f"\nTest loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")

    Path("saved_models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "test_loss": test_loss, "test_acc": test_acc,
        "all_preds": all_preds, "all_labels": all_labels, "all_genders": all_genders,
        "test_indices": test_idx,
        "binary_preds": (all_preds > 0.5).float(),
        "train_losses": train_losses, "train_accs": train_accs,
        "val_losses": val_losses,     "val_accs": val_accs,
    }, save_path)
    print(f"Modèle sauvegardé dans {save_path}")


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_training("saved_models/resnet18.pt", crop=False)
    run_training("saved_models/resnet18_crop.pt", crop=True)