import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cvt import CvT # Ensure cvt.py is in the same directory or adjust the import
import torch.nn as nn
import json

# Device configuration: GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test dataset
test_data = datasets.ImageFolder('test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Initialize the CvT model
model = CvT(image_size=224, in_channels=3, num_classes=2).to(device)

# Load the trained model
model.load_state_dict(torch.load('cvt_model.pth'))
model.eval()

# Initialize lists to collect test losses and accuracies
test_losses = []
test_accuracies = []

# Test evaluation
correct, total = 0, 0
test_loss = 0.0
batch_count = 0
batch_size = test_loader.batch_size
class_names = ['NORMAL', 'PNEUMONIA']

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        batch_count += 1

        # Print results for each image
        for i in range(inputs.size(0)):
            actual_class = class_names[labels[i].item()]
            predicted_class = class_names[predicted[i].item()]
            print(f"Image {(batch_count - 1) * batch_size + i + 1}: Actual = {actual_class}, Predicted = {predicted_class}")

        # Save results every 30 images
        if batch_count * batch_size % 32 == 0:
            test_loss_30 = test_loss / (batch_count * batch_size)
            test_accuracy_30 = 100 * correct / (batch_count * batch_size)
            test_losses.append(test_loss_30)
            test_accuracies.append(test_accuracy_30)

            print(f"Test Loss (after {batch_count * batch_size} images): {test_loss_30:.4f}")
            print(f"Test Accuracy (after {batch_count * batch_size} images): {test_accuracy_30:.2f}%")

# Final calculations for overall test loss and accuracy
test_loss = test_loss / len(test_data)
test_accuracy = 100 * correct / total

# Append final results to the lists
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

print(f'Final Test Loss: {test_loss:.4f}')
print(f'Final Test Accuracy: {test_accuracy:.2f}%')

# Load the existing data from chartData.json
try:
    with open('chartData.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    data = {
        'test_losses': [],
        'test_accuracies': []
    }

# Append the collected test losses and accuracies
data['test_losses'].extend(test_losses)
data['test_accuracies'].extend(test_accuracies)

# Save the updated data to chartData.json
with open('chartData.json', 'r+') as f:
    json.dump(data, f, indent=4)

print('Test data saved to chartData.json')