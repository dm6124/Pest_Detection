import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# ---- CONFIG ----
data_dir = r'C:\Users\Jefferson\Desktop\PROJECTS\Pest_Detection\Pest_Dataset'
save_dir = r'C:\Users\Jefferson\Desktop\PROJECTS\Pest_Detection'
batch_size = 32
num_epochs = 20
learning_rate = 0.001
num_workers = 4

# ---- CREATE SAVE DIR ----
os.makedirs(save_dir, exist_ok=True)

# ---- DEVICE ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# ---- DATA TRANSFORMS ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet input size
    transforms.ToTensor(),          # Convert images to tensors (normalize to [0,1])
])

# ---- DATASETS ----
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

# ---- HANDLE IMBALANCED DATA ----
class_counts = Counter([label for _, label in train_dataset])  # Count samples per class
print(f"Class counts: {class_counts}")
class_weights = [1.0 / class_counts[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(train_dataset), replacement=True)

# ---- DATA LOADERS ----
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=device.type == 'cuda',  
    persistent_workers=True           
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.type == 'cuda'
)

# Verify class counts match dataset labels
num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

# ---- MODEL ----
model = models.resnet18(weights='DEFAULT')  # Load ResNet-18 with pretrained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer for classification

model.to(device)  # Move model to GPU/CPU

# Multi-GPU support (if multiple GPUs are available)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# ---- LOSS & OPTIMIZER ----
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# ---- MIXED PRECISION SCALER (GPU only) ----
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# ---- TRAINING LOOP ----
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    print(f"\nStarting Epoch {epoch+1}/{num_epochs}...")
    
    # Training Phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=True)
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler:
            # Mixed precision training block (GPU only)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total     # Calculate training accuracy (%)
    train_loss = running_loss / len(train_loader)  # Average training loss
    
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation Phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():  
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total      # Validation accuracy (%)
    val_loss /= len(val_loader)                  # Average validation loss
    
    print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

# Save Model to Disk
model_path = os.path.join(save_dir, 'pest_model.pt')
torch.save(model.state_dict(), model_path)   # Save model weights only (state_dict)
print(f"✅ Model saved to {model_path}")

# Plot Training and Validation Stats
plt.figure(figsize=(10, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Over Epochs')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plot_path = os.path.join(save_dir, 'training_stats.png')
plt.tight_layout()
plt.savefig(plot_path)   # Save plots as an image file in the save directory
plt.close()

print(f"📊 Training graphs saved to {plot_path}")
