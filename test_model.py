import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cvt import CvT  # Ensure cvt.py is in the same directory or adjust the import

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

# Initialize the CvT model
model = CvT(image_size=224, in_channels=3, num_classes=2).to(device)

# Load the trained model
model.load_state_dict(torch.load('cvt_model.pth'))
model.eval()

# Test evaluation
correct, total = 0, 0
class_names = ['NORMAL', 'PNEUMONIA']

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print results for each image
        for i in range(inputs.size(0)):
            actual_class = class_names[labels[i].item()]
            predicted_class = class_names[predicted[i].item()]
            print(f"Image {i+1}: Actual = {actual_class}, Predicted = {predicted_class}")

# Overall test accuracy
print(f'Test Accuracy: {100 * correct / total:.2f}%')