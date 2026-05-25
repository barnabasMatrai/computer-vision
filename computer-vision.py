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
# In this section, we define the hyperparameters used for training.
# These include the batch size, learning rate, number of epochs,
# and the train/test split ratio.
# We also set a random seed for reproducibility and select the device.

SEED = 42
TEST_RATIO = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

# ---------------- DATA LOADING ----------------
# Load original dataset without preprocessing
dataset = datasets.ImageFolder(root="images")

# Class names
class_names = dataset.classes

# Total number of classes
num_classes = len(class_names)

# ---------------- PREPROCESSING ----------------
'''
PREPROCESSING STEPS
1. Resize all images to 224x224 
- CNN models require the same input size 
2. Convert images into tensors 
- PyTorch works with tensor data 
3. Normalize pixel values 
- Makes training more stable 
4. Apply data augmentation 
- Helps reduce overfitting 
- Creates more diverse training data 
5. Apply random rotation 
- Helps the model learn different angles 
6. Apply random translation 
- Helps the model recognize shifted objects 
7. Apply random crop 
- Helps the model focus on different parts of the object 
8. Split dataset into training and testing sets 
- Training set for learning - Test set for evaluation 
9. Create DataLoaders 
- Loads images in batches during training 
10. Verify image shapes and labels 
- Ensure preprocessing works correctly
'''
# Image preprocessing - train uses augmentation, test stays deterministic
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # Random horizontal flip
    transforms.RandomHorizontalFlip(),
   # Random rotation
    transforms.RandomRotation(15),

   # Random crop
    transforms.RandomResizedCrop(
        256,
        scale=(0.8, 1.0)
    ),
    # Convert image to tensor
    transforms.ToTensor(),

     # Normalize pixel values
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

# Test preprocessing pipeline
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

# Visualization preprocessing (without normalization)
# Normalization is not applied here because it would make
# the images appear darker during visualization.
visualization_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),

    transforms.RandomResizedCrop(
        256,
        scale=(0.8, 1.0)
    ),

    transforms.ToTensor(),
])

# ---------------- APPLY PREPROCESSING TO DATASETS ----------------
# Two views of the same folder so train and test get different transforms
train_dataset = datasets.ImageFolder(root="images", transform=train_transform)
test_dataset = datasets.ImageFolder(root="images", transform=test_transform)
num_classes = len(train_dataset.classes)

# ---------------- PREPROCESSING VISUALIZATION ----------------
fig, axes = plt.subplots(4, 2, figsize=(10,16))

for i in range(4):

    # Select class
    class_name = class_names[i]

    # Select image
    image_path = os.path.join("images",class_name,random.choice(os.listdir(os.path.join("images", class_name))))

    # Open image
    original_image = Image.open(image_path)

    # Apply preprocessing
    processed_image = visualization_transform(original_image)

    # Convert tensor for matplotlib
    processed_image = processed_image.permute(1, 2, 0)

    # Original image
    axes[i, 0].imshow(original_image)
    axes[i, 0].set_title(f"{class_name} - Before")
    axes[i, 0].axis("off")

    # Processed image
    axes[i, 1].imshow(processed_image)
    axes[i, 1].set_title(f"{class_name} - After")
    axes[i, 1].axis("off")

plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.show()

# ---------------- TRAIN / TEST SPLIT ----------------
# Stratified train/test split (same indices applied to both views)
indices = list(range(len(train_dataset)))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_RATIO,
    stratify=train_dataset.targets,
    random_state=SEED,
)

train_set = Subset(train_dataset, train_idx)
test_set = Subset(test_dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Verify image shapes and labels
images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
# ----------------  DATA VISUALIZATION ----------------
# Get all class folders
# We filtered hidden system files from the dataset directory, because macOS automatically creates files such as .DS_Store, which are not actual class folders.
class_folders = [folder for folder in os.listdir("images") if not folder.startswith(".")]

#We used a 5x5 grid to display one image from each class.
# create 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(8,8))

for i in range(len(class_folders)):

    # get current class name
    class_name = class_folders[i]

    # path to class folder
    class_folder_path = os.path.join("images", class_name)

    # get first image in folder
    image_name = os.listdir(class_folder_path)[0]

    # full image path
    image_path = os.path.join(class_folder_path, image_name)

    # open image
    image = Image.open(image_path)

    # calculate row and column
    row = i // 5
    col = i % 5

    # show image
    axes[row, col].imshow(image)

    # set title
    axes[row, col].set_title(class_name)

plt.tight_layout()
plt.show()


# ---------------- DATASET ANALYSIS ----------------
"""
In this section, we analyze the Freiburg Groceries Dataset to better understand.

First, we calculate the total number of images and classes.
Then, we inspect all class names and count how many images
exist in each category.

-> This helps us understand whether the dataset is balanced
or if some classes contain significantly more images than others.

We also check the image resolution and visualize the class
distribution using a bar chart.
"""

# print total number of images
print("Total images:", len(train_dataset))

# print total number of classes
print("Number of classes:", len(train_dataset.classes))

# print all class names
print("Class names:")
print(train_dataset.classes)

# count images in each class
print("\nImages per class:")

class_counts = []

for class_name in train_dataset.classes:

    # create path to class folder
    class_path = os.path.join("images", class_name)

    # count images inside the folder
    image_count = len(os.listdir(class_path))

    # save image count for graph
    class_counts.append(image_count)

    # print class and image count
    print(class_name, ":", image_count)

# open one sample image
sample_class = train_dataset.classes[0]

sample_folder = os.path.join("images", sample_class)

sample_image_name = os.listdir(sample_folder)[0]

sample_image_path = os.path.join(
    sample_folder,
    sample_image_name
)

sample_image = Image.open(sample_image_path)

# print image resolution
print("\nImage resolution:", sample_image.size)


# ---------------- CLASS DISTRIBUTION GRAPH ----------------

"""
In this section, we display one sample image from each class.
"""
plt.figure(figsize=(12,6))

# create bar chart
plt.bar(train_dataset.classes, class_counts)

# rotate class names
plt.xticks(rotation=90)

# graph title
plt.title("Number of Images per Class")

# axis labels
plt.xlabel("Class Name")
plt.ylabel("Image Count")

# show graph
plt.show()


# ---------------- CNN MODEL ----------------
# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
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

# ---------------- LOSS FUNCTION AND OPTIMIZER ----------------
model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ---------------- TRAINING LOOP ----------------
# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    # --- train ---
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

    # --- eval ---
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
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"epoch {epoch}/{NUM_EPOCHS} | "
          f"train loss {train_loss / train_total:.4f} "
          f"train acc {train_correct / train_total:.3f} | "
          f"test loss {avg_test_loss:.4f} "
          f"test acc {test_correct / test_total:.3f} | "
          f"lr {current_lr:.1e}")

