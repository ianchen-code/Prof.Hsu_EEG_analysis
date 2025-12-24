import mne
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========================================
# 1. LOAD AND ORGANIZE DATA
# ========================================
cropped_folder = '/Volumes/USB/EEG research/Data3/cropped_segments'

# Separate files by label
correct_files = []  # trigger 201
incorrect_files = []  # trigger 200

for filename in os.listdir(cropped_folder):
    # Skip hidden files (macOS ._* files) and non-.fif files
    if filename.startswith('._') or filename.startswith('.'):
        continue
    if filename.endswith('.fif'):
        if 'trig' in filename:
            # Extract trigger info from filename
            if '-201.fif' in filename:
                correct_files.append(os.path.join(cropped_folder, filename))
            elif '-200.fif' in filename:
                incorrect_files.append(os.path.join(cropped_folder, filename))

print(f"Correct trials (201): {len(correct_files)}")
print(f"Incorrect trials (200): {len(incorrect_files)}")

# Handle class imbalance - undersample majority class
if len(correct_files) > len(incorrect_files):
    print(f"\nClass imbalance detected! Undersampling majority class...")
    np.random.seed(42)
    correct_files = list(np.random.choice(correct_files, size=len(incorrect_files), replace=False))
    print(f"After balancing - Correct: {len(correct_files)}, Incorrect: {len(incorrect_files)}")
elif len(incorrect_files) > len(correct_files):
    print(f"\nClass imbalance detected! Undersampling majority class...")
    np.random.seed(42)
    incorrect_files = list(np.random.choice(incorrect_files, size=len(correct_files), replace=False))
    print(f"After balancing - Correct: {len(correct_files)}, Incorrect: {len(incorrect_files)}")

# Load all data
def load_eeg_segment(filepath):
    try:
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        data = raw.get_data()  # Shape: (n_channels, n_samples)
        return data
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None

X_correct = [load_eeg_segment(f) for f in correct_files]
X_incorrect = [load_eeg_segment(f) for f in incorrect_files]

# Filter out None values (failed loads)
X_correct = [x for x in X_correct if x is not None]
X_incorrect = [x for x in X_incorrect if x is not None]

print(f"Successfully loaded - Correct: {len(X_correct)}, Incorrect: {len(X_incorrect)}")

# Check shapes and find length distribution
all_lengths = [x.shape[1] for x in X_correct + X_incorrect]
all_channels = [x.shape[0] for x in X_correct + X_incorrect]
print(f"\nData statistics:")
print(f"  Channels - Min: {np.min(all_channels)}, Max: {np.max(all_channels)}, Unique: {np.unique(all_channels)}")
print(f"  Samples - Min: {np.min(all_lengths)}, Max: {np.max(all_lengths)}, Mean: {np.mean(all_lengths):.1f}")

# Choose fixed dimensions
fixed_channels = int(np.min(all_channels))  # Use minimum to avoid adding fake channels
fixed_length = int(np.median(all_lengths))
print(f"  Using fixed dimensions: {fixed_channels} channels × {fixed_length} samples")

# Standardize both channels and length
def standardize_data(data, target_channels, target_length):
    """Standardize number of channels and length"""
    n_channels, n_samples = data.shape
    
    # First, standardize channels
    if n_channels > target_channels:
        # Keep first target_channels (usually EEG channels come first)
        data = data[:target_channels, :]
    elif n_channels < target_channels:
        # This shouldn't happen if we use min, but handle it anyway
        pad_width = ((0, target_channels - n_channels), (0, 0))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
    
    # Then, standardize length
    n_channels, n_samples = data.shape  # Update after channel adjustment
    
    if n_samples > target_length:
        # Crop from center
        start = (n_samples - target_length) // 2
        return data[:, start:start + target_length]
    elif n_samples < target_length:
        # Pad with zeros
        pad_width = ((0, 0), (0, target_length - n_samples))
        return np.pad(data, pad_width, mode='constant', constant_values=0)
    else:
        return data

X_correct = [standardize_data(x, fixed_channels, fixed_length) for x in X_correct]
X_incorrect = [standardize_data(x, fixed_channels, fixed_length) for x in X_incorrect]

# Verify all have same shape
print(f"\nAfter standardization:")
print(f"  Correct samples shapes: {set([x.shape for x in X_correct])}")
print(f"  Incorrect samples shapes: {set([x.shape for x in X_incorrect])}")

# Now convert to arrays (all same shape)
X_correct = np.array(X_correct)  # (n_correct, fixed_channels, fixed_length)
X_incorrect = np.array(X_incorrect)  # (n_incorrect, fixed_channels, fixed_length)

# Create labels
y_correct = np.ones(len(X_correct))   # 1 for correct
y_incorrect = np.zeros(len(X_incorrect))  # 0 for incorrect

# Combine
X = np.concatenate([X_correct, X_incorrect], axis=0)
y = np.concatenate([y_correct, y_incorrect], axis=0)

print(f"\nTotal data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Class distribution - Correct: {np.sum(y)}, Incorrect: {len(y) - np.sum(y)}")

# ========================================
# 2. TRAIN-TEST SPLIT
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# ========================================
# 3. EEGNET MODEL (Pretrained Architecture)
# ========================================
class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_samples=800, n_classes=2, F1=8, F2=16, D=2):
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise convolution (spatial filtering)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), 
                                   groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.5)
        
        # Block 3: Separable convolution
        self.separable1 = nn.Conv2d(F1 * D, F2, (1, 16), 
                                    padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layer
        self.flatten = nn.Flatten()
        # Calculate the size after convolutions
        self.fc_input_size = F2 * (n_samples // 32)  # Adjusted for pooling
        self.fc = nn.Linear(self.fc_input_size, n_classes)
        
    def forward(self, x):
        # Input: (batch, 1, channels, samples)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.separable1(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ========================================
# 4. PREPARE DATA FOR PYTORCH
# ========================================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dim
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EEGDataset(X_train, y_train)
test_dataset = EEGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========================================
# 5. TRAINING - Apple Silicon GPU Support
# ========================================
# Check for Apple Silicon MPS (Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"\nUsing device: Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"\nUsing device: NVIDIA GPU (CUDA)")
else:
    device = torch.device('cpu')
    print(f"\nUsing device: CPU")

print(f"PyTorch version: {torch.__version__}")

n_channels = X.shape[1]
n_samples = X.shape[2]
model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                        factor=0.5, patience=5)

# Training loop
n_epochs = 50
best_acc = 0

print("\nStarting training...")
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    
    # Validation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    
    # Update learning rate scheduler
    scheduler.step(test_acc)
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_eegnet_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Test Acc: {test_acc:.2f}%")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")
print("Model saved as 'best_eegnet_model.pth'")

# ========================================
# 6. EVALUATION
# ========================================
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model.load_state_dict(torch.load('best_eegnet_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, 
                          target_names=['Incorrect (200)', 'Correct (201)']))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Incorrect', 'Correct'],
            yticklabels=['Incorrect', 'Correct'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")