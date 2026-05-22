from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

# ---------------- DATA LOADING ----------------

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(
    root="images",
    transform=transform
)

# Create batches
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Class names
print(dataset.classes)

# Example batch
images, labels = next(iter(loader))

print(images.shape)   # torch.Size([32, 3, 224, 224])
print(labels.shape)



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
print("Total images:", len(dataset))

# print total number of classes
print("Number of classes:", len(dataset.classes))

# print all class names
print("Class names:")
print(dataset.classes)

# count images in each class
print("\nImages per class:")

class_counts = []

for class_name in dataset.classes:

    # create path to class folder
    class_path = os.path.join("images", class_name)

    # count images inside the folder
    image_count = len(os.listdir(class_path))

    # save image count for graph
    class_counts.append(image_count)

    # print class and image count
    print(class_name, ":", image_count)

# open one sample image
sample_class = dataset.classes[0]

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
plt.bar(dataset.classes, class_counts)

# rotate class names
plt.xticks(rotation=90)

# graph title
plt.title("Number of Images per Class")

# axis labels
plt.xlabel("Class Name")
plt.ylabel("Image Count")

# show graph
plt.show()

# ---------------- PREPROCESSING ----------------

"""
PREPROCESSING TO DO LIST

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
   - Training set for learning
   - Test set for evaluation

9. Create DataLoaders
   - Loads images in batches during training

10. Verify image shapes and labels
   - Ensure preprocessing works correctly

"""