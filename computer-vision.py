import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import ImageFilter
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from sklearn.metrics import confusion_matrix

# ---------------- CONFIGURATION ----------------
# In this section, we define the hyperparameters used for training.
# These include the batch size, learning rate, number of epochs,
# and the train/test split ratio.
# We also set a random seed for reproducibility and select the device.


SEED = 42
# %80 training, %20 testing
TEST_RATIO = 0.2
# batch size defines how many images are processed together in one training step.
BATCH_SIZE = 16
# epoch means one complete pass through the training dataset.
NUM_EPOCHS = 50
# learning rate controls how strongly the model updates its weights during training.
LEARNING_RATE = 1e-3

torch.manual_seed(SEED)

#it selects which hardware device will be used for training the CNN model.
# code first checks whether MPS is available. If MPS is available, the model uses the Apple GPU for faster training.
# otherwise, the code switches to CUDA, which is NVIDIA’s GPU acceleration platform commonly used on Windows and Linux systems with NVIDIA graphics cards.
# using GPU acceleration is important because CNN training requires many mathematical operations, and GPUs can process these operations much faster than CPUs.
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"device: {device}")

# ---------------- DATA LOADING ----------------

# Load original dataset without preprocessing
# ImageFolder function reads all subfolders inside the images directory
# and uses folder names as class labels
# ImageFolder is provided by the torchvision library.
dataset = datasets.ImageFolder(root="images")

# .classes returns the list of category names found in the dataset.
class_names = dataset.classes


# ---------------- PREPROCESSING ----------------
# !!!The preprocessing pipeline is separated into two different parts: train_transform and test_transform.
'''
train_transform contains data augmentation operations 
such as RandomRotation, RandomResizedCrop and RandomAffine. 
These transformations randomly modify the training images by rotating, cropping and shifting them. 
This helps create more diverse training data and reduces overfitting.
'''
# Image preprocessing - train uses augmentation, test stays deterministic
# Compose is used to apply several image transformations step by step.
train_transform = transforms.Compose([
    # RESIZE standardizes image dimensions before training the CNN model
    transforms.Resize((256, 256)),

    # ROTATION
    #randomly rotate the image between -15 and +15
    #One time the image might be rotated: +8 (clockwise) and another time: -12 (counterclockwise)
    #So the same image can appear differently in different epochs.
    #Every epoch, the same image can be loaded again with a different random rotation. For example, the image might be rotated by +5 in one epoch and by -12 in another epoc
    transforms.RandomRotation(15),

    # CROP
    # Randomly selects and crops a region of the image
    # between 80% and 100% of the original size,
    # then resizes it to 256x256 pixels.
    #When an image is resized, the number of pixels changes. Because of this, the model needs a method to calculate the values of the new pixels and uses bilinear interpolation for this calculation.
    #Bilinear interpolation calculates new pixel values by looking at the neighboring pixels around them
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0),interpolation=InterpolationMode.BILINEAR),

    # TRANSLATION (SHIFT)
    # degrees=0 means that no rotation is applied. The image is not rotated and only translation is performed.
    # translate=(0.1, 0.1) = image can be randomly shifted by up to 10% of its width and height.
    # For example, with a 256×256 image: 256 × 0.1 ≈ 25 pixels 
    # So the image may be shifted: left or right by about 25 pixels OR up or down by about 25 pixels.
    # The transformation is random and can change every epoch.
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    
    #SHARPEN
    # This transformation sharpens the image before it is converted into a tensor.
    # ImageFilter.SHARPEN is a PIL image filter that increases edge contrast and makes details appear clearer and more defined. For example, object borders and textures may become slightly more visible after sharpening.
    # The transformation is wrapped inside transforms.Lambda(...)
     #because PyTorch does not provide a built-in sharpen transform directly in torchvision.transforms
    transforms.Lambda(lambda img: img.filter(ImageFilter.SHARPEN)),

    # convert to tensor
    # At the beginning, the image is a normal PIL image object created by:Image.open(...)
    # A PIL image is mainly used for image processing operations such as:filtering, cropping, resizing, etc.
    # However, a CNN cannot work directly with a PIL image. Neural networks require numerical data that can be processed mathematically.
    # ToTensor() mainly does two things:
    # 1. converts the image into a PyTorch tensor,
    # 2. scales the pixel values from 0–255 to the range 0.0–1.0.
    # and the CNN can use for matrix multiplications, forward and backward propagation.
    # After ToTensor(), the image becomes a multidimensional numerical array called a tensor.
    # for exampe an RGB image becomes: [3, 256, 256]
    # grayscale image may become: [256, 256]
    transforms.ToTensor(),

    # normalize
    # Normalization is applied after ToTensor() because normalization requires numerical tensor values, not a normal PIL image.
    # It may look unnecessary at first because ToTensor() already scales the pixel values to the range 0.0–1.0.
    # However, normalization is still useful because CNNs usually train more efficiently when the input values are centered around zero rather than containing only positive values.
    #ToTensor() -> scales values
    #Normalize() -> centers and standardizes values
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #An RGB image tensor usually has this structure: [3, height, width]
    # PyTorch normalizes each channel separately. So Normalize(mean=[R,G,B], std=[R,G,B])
    #channel 0 → Red , channel 1 → Green , channel 2 → Blue
    
])

# Test preprocessing pipeline


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
])

# Visualization preprocessing (without normalization)
# Normalization is not applied here because it would make
# the images appear darker during visualization.
visualization_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomRotation(15),

    transforms.RandomResizedCrop(
        256,
        scale=(0.8, 1.0),
        interpolation=InterpolationMode.BILINEAR
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

    transforms.Lambda(lambda img: img.filter(ImageFilter.SHARPEN)),

    transforms.ToTensor(),
])

# ---------------- APPLY PREPROCESSING TO DATASETS ----------------
'''
Why are there two ImageFolder datasets?
Because:
- train should receive augmentation
- test should not receive augmentation
Then, train_idx and test_idx determine which images belong to the training set and which belong to the test set.
'''

train_dataset = datasets.ImageFolder(root="images", transform=train_transform)
'''
means:
Read the images folder
and apply the train_transform pipeline
as each image is loaded
'''
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
'''
Even though both ImageFolder datasets initially read have ALL images, train_test_split() creates different index lists.
Each image index is assigned either to train_idx or to test_idx, not both.
After that, Subset uses these indices to create separate train and test sets.
So the same image cannot appear in both the training set and the test set.
'''
#creates all image numbers: [0, 1, 2, 3]
indices = list(range(len(train_dataset)))

#splits these numbers:
#train_idx = [0, 1]
#test_idx = [2, 3]
train_idx, test_idx = train_test_split(indices,test_size=TEST_RATIO,stratify=train_dataset.targets,random_state=SEED,)

#means: use train_dataset, but only images with train_idx. These images get train_transform.
train_set = Subset(train_dataset, train_idx)
test_set = Subset(test_dataset, test_idx)

#DataLoader is a PyTorch utility that loads the dataset and sends the data to the CNN during the training and testing phases.
#DataLoader is responsible for:loading images, creating batches, and sending the batches to the CNN during training/testing.
#If the training set contains 320 images and: batch_size = 32 then 320 / 32 = 10 batches per epoch.  CNN processes one batch at a time.
#Why is shuffle=True used for training? it randomly changes the order of the training images every epoch. This is important because if the CNN always sees images in the exact same order, it may learn order-related patterns
# and because
# gradient is calculated based on the batch. If the batch consistently consists of the same class, the gradient may be biased in the direction of that class.
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
#During testing, randomness is unnecessary because:weights are no longer updated, gradients are not calculated, and CNN is only performing evaluation.
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

sample_image_path = os.path.join(sample_folder, sample_image_name)

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
# ---------------- TRAINING HISTORY ----------------

train_acc_history = []
test_acc_history = []


# ---------------- LOSS FUNCTION AND OPTIMIZER ----------------
model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ---------------- EARLY STOPPING ----------------

best_loss = float("inf")

patience_counter = 0
#If the validation/test loss does not improve for 5 consecutive epochs, the training process is stopped.
PATIENCE = 5

# ---------------- TRAINING LOOP ----------------
all_predictions = []
all_labels = []
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
            predictions = logits.argmax(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = criterion(logits, labels)
            test_loss += loss.item() * images.size(0)
            test_correct += (logits.argmax(1) == labels).sum().item()
            test_total += images.size(0)

    avg_test_loss = test_loss / test_total

    # ---------------- EARLY STOPPING CHECK ----------------

    if avg_test_loss < best_loss:

        best_loss = avg_test_loss
        patience_counter = 0

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("Early stopping triggered.")

        break
    
    scheduler.step(avg_test_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"epoch {epoch}/{NUM_EPOCHS} | "
          f"train loss {train_loss / train_total:.4f} "
          f"train acc {train_correct / train_total:.3f} | "
          f"test loss {avg_test_loss:.4f} "
          f"test acc {test_correct / test_total:.3f} | "
          f"lr {current_lr:.1e}")
    # Save metrics for visualization
    train_acc_history.append(train_correct / train_total)
    test_acc_history.append(test_correct / test_total)
    



# ---------------- ACCURACY GRAPH ----------------
#accuracy graph shows how the model performance changes during training over multiple epochs.
#if both training and test accuracy improve over time, we can say, model is learning successfully.
#if the training accuracy continues to increase while the test accuracy stops improving or decreases, this may indicate overfitting.

plt.figure(figsize=(10,5))

epochs = range(1, NUM_EPOCHS + 1)

plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(test_acc_history, label="Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training and Test Accuracy")

plt.legend()

plt.show()

# ---------------- CONFUSION MATRIX ----------------
# cm helps identify which classes are predicted correctly and which classes are not
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(12,10))

plt.imshow(cm, cmap="Blues")

plt.xticks(range(len(class_names)), class_names, rotation=90)
plt.yticks(range(len(class_names)), class_names)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.colorbar()

plt.show()