import torchvision.models as models
import torch.nn as nn
import torch
import pandas as pd
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm,trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class CelebADataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        label  = torch.tensor((row["Smiling"] + 1) / 2, dtype=torch.float32)
        gender = torch.tensor((row["Male"] + 1) / 2, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label, gender


BASE_DIR = os.getcwd()
CELEBA_DIR = os.path.join(BASE_DIR, "data", "CelebA")

dataset = CelebADataset(
    csv_file=os.path.join(CELEBA_DIR, "list_attr_celeba.csv"),
    img_dir=os.path.join(CELEBA_DIR, "img_align_celeba/img_align_celeba"),
    transform=transform
)

batch_size = 128

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

torch.manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)


def make_loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4, 
        pin_memory=True,
         persistent_workers=True
    )

if __name__ == "__main__":

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset,   shuffle=False)
    test_loader = make_loader(test_dataset,  shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)


    def train_epoch(model, loader):
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(loader, desc="Training", leave=False)
        for imgs, labels, genders in pbar:
            imgs   = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((preds.sigmoid() > 0.5) == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        return total_loss / len(loader), correct / total


    def eval_model(model, loader):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels, all_genders = [], [], []

        pbar = tqdm(loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for imgs, labels, genders in pbar:
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1)

                preds = model(imgs)
                total_loss += criterion(preds, labels).item()
                correct += ((preds.sigmoid() > 0.5) == labels).sum().item()
                total += labels.size(0)

                all_preds.append(preds.sigmoid().cpu())
                all_labels.append(labels.cpu())
                all_genders.append(genders.cpu())

                pbar.set_postfix(loss=f"{total_loss/len(loader):.4f}", acc=f"{correct/total:.4f}")

        all_preds = torch.cat(all_preds).squeeze()
        all_labels = torch.cat(all_labels).squeeze()
        all_genders = torch.cat(all_genders).squeeze()

        return total_loss / len(loader), correct / total, all_preds, all_labels, all_genders

    epochs = 5

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for e in trange(epochs, desc="Epochs"):
        train_loss, train_acc = train_epoch(model, train_loader)
        val_loss, val_acc, _, _, _ = eval_model(model, val_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {e+1}/{epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val   loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

    test_loss, test_acc, all_preds, all_labels, all_genders = eval_model(model, test_loader)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    MODEL_DIR = Path("saved_models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / "resnet18.pt"

    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs":     epochs,
        "batch_size": batch_size,
        "test_loss":  test_loss,
        "test_acc":   test_acc,
        "all_preds":    all_preds,
        "all_labels":   all_labels,
        "all_genders":  all_genders,
        "test_indices": test_dataset.indices,  
        "binary_preds": (all_preds > 0.5).float(),
        "train_losses": train_losses,
        "train_accs":   train_accs,
        "val_losses":   val_losses,
        "val_accs":     val_accs,
    }, save_path)

    print(f"Modèle sauvegardé dans {save_path}")