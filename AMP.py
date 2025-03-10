import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models

# Check if a GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (required for ResNet)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet values
])

# Load CIFAR-10 dataset (training data)
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer to match the number of classes (CIFAR-10 has 10 classes)
model.fc = nn.Linear(model.fc.in_features, 10)

# Move the model to the selected device (GPU/CPU)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropy is used for classification problems
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Initialize the GradScaler (new method)
scaler = torch.amp.GradScaler()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Track the total loss for this epoch
    
    for images, labels in train_loader:
        # Move images and labels to the selected device
        images, labels = images.to(device), labels.to(device)

        # Reset the gradients
        optimizer.zero_grad()

        # Use automatic mixed precision (AMP) for faster training
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

        # Scale the loss for stable training
        scaler.scale(loss).backward()  # Backpropagation
        scaler.step(optimizer)  # Update model weights
        scaler.update()  # Update the scaler for the next iteration

        # Accumulate loss
        running_loss += loss.item()

    # Calculate and print the average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print(" Training completed successfully!")
