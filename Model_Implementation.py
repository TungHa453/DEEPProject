import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None, max_samples=100):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []

        for label in ['NORMAL', 'PNEUMONIA']:
            label_dir = os.path.join(directory, label)
            for img_file in os.listdir(label_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(label_dir, img_file))
                    self.labels.append(0 if label == 'NORMAL' else 1)  # 0 for normal, 1 for pneumonia
                    # Stop if we've reached the maximum number of samples

                    if len(self.images) >= max_samples:
                        break
                if len(self.images) >= max_samples:
                    break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# Image Transformation Function
def transform_image(image):
    image = image.resize((224, 224))  # Resize image
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Change to CxHxW
    return image


# Define CVT Classes (same as previous)
class CVTEmbedding(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, stride):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)
        x = x.flatten(2).permute(0, 2, 1)  # Flatten and permute for LayerNorm
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))
        attn_weights = attn_scores.softmax(dim=-1)
        return (attn_weights @ v).reshape(B, N, C)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class CvT(nn.Module):
    def __init__(self, embed_dim, num_class):
        super().__init__()
        self.embedding = CVTEmbedding(in_ch=3, embed_dim=embed_dim, patch_size=7, stride=4)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads=4) for _ in range(6)])
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


# Function to train the model
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    print('Training complete.')


# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Function to load an image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform_image(image).unsqueeze(0).to(device)  # Add batch dimension


# Load images from a directory for prediction
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            image_tensor = load_image(image_path)
            images.append((filename, image_tensor))
    return images


# Predict all images in a directory
def predict_all_images(model, image_directory):
    model.eval()
    predictions = {}
    images = load_images_from_directory(image_directory)

    with torch.no_grad():
        for filename, image_tensor in images:
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            predictions[filename] = predicted.item()

    return predictions


if __name__ == '__main__':
    # Parameters
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 2  # Binary classification: normal and pneumonia
    dataset_path = 'train'  # Update this path to your dataset directory
    image_directory = 'test'  # Update this path to the directory with images to predict

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data Preparation
    dataset = CustomImageDataset(directory=dataset_path, transform=transform_image, max_samples=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = CvT(embed_dim=64, num_class=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)

    # Save the model
    save_model(model, 'cvt_model.pth')

    # Load the model for prediction
    model.load_state_dict(torch.load('cvt_model.pth'))

    # Predict all images in a directory
    predictions = predict_all_images(model, image_directory)

    # Print predictions for each image
    for filename, predicted_class in predictions.items():
        print(f'Image: {filename}, Predicted class: {predicted_class}')