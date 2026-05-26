import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image
import os
import random


# ---------------- CONFIGURATION ----------------
SEED = 42
TEST_RATIO = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

torch.manual_seed(SEED)


# ---------------- MODEL ----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------- MAIN ----------------
if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # ---------------- DATA LOADING ----------------
    dataset = datasets.ImageFolder(root="images")
    class_names = dataset.classes
    num_classes = len(class_names)

    # ---------------- TRANSFORMS ----------------
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    visualization_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    # ---------------- DATASETS ----------------
    train_dataset = datasets.ImageFolder(root="images", transform=train_transform)
    test_dataset = datasets.ImageFolder(root="images", transform=test_transform)

    indices = list(range(len(train_dataset)))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_RATIO,
        stratify=train_dataset.targets,
        random_state=SEED,
    )

    train_set = Subset(train_dataset, train_idx)
    test_set = Subset(test_dataset, test_idx)

    # ---------------- DATALOADERS ----------------
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    # ---------------- QUICK CHECK ----------------
    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    # ---------------- MODEL ----------------
    model = SimpleCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ---------------- TRAINING ----------------
    for epoch in range(1, NUM_EPOCHS + 1):

        # ---- train ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        # ---- eval ----
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                test_loss += loss.item() * images.size(0)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += images.size(0)

        avg_test_loss = test_loss / test_total
        scheduler.step(avg_test_loss)

        print(
            f"epoch {epoch}/{NUM_EPOCHS} | "
            f"train acc {train_correct/train_total:.3f} | "
            f"test acc {test_correct/test_total:.3f}"
        )