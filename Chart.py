import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cvt import CvT
import matplotlib.pyplot as plt
import json

# Device configuration: GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load validation dataset
val_data = datasets.ImageFolder('val', transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize the CvT model
model = CvT(image_size=224, in_channels=3, num_classes=2).to(device)

# Load the trained model
model.load_state_dict(torch.load('cvt_model.pth'))
model.eval()

with open('chartData.json', 'r') as f:
    data = json.load(f)

# Get the train and validation metrics from the loaded dictionary
train_losses = data.get('train_losses', [])
train_accuracies = data.get('train_accuracies', [])
val_losses = data.get('val_losses', [])
val_accuracies = data.get('val_accuracies', [])
test_losses = data.get('test_losses', [])  # Removed the extra comma
test_accuracies = data.get('test_accuracies', [])  # Removed the extra comma

# Plot the training, validation, and test data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss')  # No need for list multiplication
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.legend()

plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')  # No need for list multiplication
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()