from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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