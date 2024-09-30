import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json

from cvt import CvT

# Device configuration: GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_data = datasets.ImageFolder('val', transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize the CvT model
model = CvT(image_size=224, in_channels=3, num_classes=2).to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Validation step
model.load_state_dict(torch.load('cvt_model.pth'))
model.eval()

val_losses = []
val_accuracies = []

# Test Validation
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        val_losses.append(loss.item())  # Store loss for this batch

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        batch_accuracy = 100 * correct / labels.size(0)  # Accuracy for this batch
        val_accuracies.append(batch_accuracy)  # Store accuracy for this batch

# Print average metrics
avg_val_loss = sum(val_losses) / len(val_losses)
avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)

print(f'Average Validation Loss: {avg_val_loss:.4f}')
print(f'Average Validation Accuracy: {avg_val_accuracy:.2f}%')

# Load the chart data from chartData.json
try:
    with open('chartData.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    data = {
        'val_losses': [],
        'val_accuracies': []
    }

# Update the chartData.json with validation data
data['val_losses'].extend(val_losses)
data['val_accuracies'].extend(val_accuracies)

with open('chartData.json', 'w') as f:
    json.dump(data, f, indent=4)

print('Validation metrics saved to chartData.json')