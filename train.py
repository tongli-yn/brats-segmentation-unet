
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import Compose, Resize, ScaleIntensityRange, ToTensor
from monai.losses import DiceLoss
from models import get_model
from datasets.brats_dataset import BraTSDataset2D
from utils.metrics import dice_score

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def check(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_model(config):
    # Load data
    import pandas as pd
    meta_df = pd.read_csv(config["data_csv"])

    image_transform = Compose([
        ScaleIntensityRange(0, 1, 0.0, 1.0, clip=True),
        Resize(config["image_size"]),
        ToTensor()
    ])
    label_transform = Compose([
        Resize(config["image_size"], mode="nearest"),
        ToTensor()
    ])

    dataset = BraTSDataset2D(meta_df, image_transform, label_transform)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    # Model
    model = get_model(config["model_name"], config["in_channels"], config["out_channels"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    writer = SummaryWriter("runs/exp1")
    early_stopper = EarlyStopping(patience=config["patience"])

    best_dice = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = ce_loss(pred, y) + dice_loss(pred, y.unsqueeze(1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        writer.add_scalar("Loss/Train", epoch_loss / len(train_loader), epoch)

        # Validation
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = torch.argmax(model(x), dim=1)
                dices.append(np.mean(dice_score(pred, y)))
        avg_dice = np.mean(dices)
        writer.add_scalar("Dice/Val", avg_dice, epoch)
        print(f"✅ Val Dice: {avg_dice:.4f}")

        scheduler.step(avg_dice)

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), f"best_{config['model_name']}_unet.pth")
            print("💾 Best model saved.")

        if early_stopper.check(avg_dice):
            print("🛑 Early stopping triggered.")
            break

    writer.close()


if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    train_model(config)
