import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
from cvt import CvT  # Ensure cvt.py is in the same directory or adjust the import
import json


# Device configuration: GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_data = datasets.ImageFolder('train', transform=transform)
val_data = datasets.ImageFolder('val', transform=transform)

# Subset the train dataset to only use 1000 images
indices = random.sample(range(len(train_data)), 1000)
train_data = Subset(train_data, indices)

# DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize the CvT model
model = CvT(image_size=224, in_channels=3, num_classes=2).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
train_accuracies = []
train_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_data)
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# Save the model
torch.save(model.state_dict(), 'cvt_model.pth')
print("Model saved as cvt_model.pth")

# Save the training datas
chartData = {
    'train_losses': train_losses,
    'train_accuracies': train_accuracies
}

with open('chartData.json', 'w') as f:
    json.dump(chartData, f, indent=4)

print('Trained data saved to chartData.json')