import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.onnx
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Device configuration (Use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001


# --- CUSTOM DATASET CLASS TO LOAD YOUR LOCAL UBYTE FILES ---
class LocalFashionMNIST(Dataset):
    def __init__(self, root_dir, kind='train', transform=None):
        """
        Reads ubyte files from the specified 'Ubytes' folder.
        root_dir: Path to the 'Ubytes' folder.
        kind: 'train' for training data, 't10k' for test data.
        """
        self.transform = transform

        # Define paths to your specific files
        if kind == 'train':
            img_path = os.path.join(root_dir, 'train-images-idx3-ubyte')
            lbl_path = os.path.join(root_dir, 'train-labels-idx1-ubyte')
        else:
            img_path = os.path.join(root_dir, 't10k-images-idx3-ubyte')
            lbl_path = os.path.join(root_dir, 't10k-labels-idx1-ubyte')

        # Load images
        with open(img_path, 'rb') as f_img:
            # Skip the 16-byte header (magic number, num images, rows, cols)
            # np.frombuffer reads the rest as a flat array
            data = np.frombuffer(f_img.read(), dtype=np.uint8, offset=16)

        # Load labels
        with open(lbl_path, 'rb') as f_lbl:
            # Skip the 8-byte header (magic number, num labels)
            labels = np.frombuffer(f_lbl.read(), dtype=np.uint8, offset=8)

        # Reshape from (Total_Pixels) to (Num_Images, 28, 28)
        self.images = data.reshape(-1, 28, 28)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image as numpy array
        img = self.images[idx]

        # Add channel dimension: (28, 28) -> (28, 28, 1)
        # Required because transforms usually expect (H, W, C)
        img = np.expand_dims(img, axis=2)

        if self.transform:
            img = self.transform(img)

        label = int(self.labels[idx])
        return img, label


# 1. Data Loading
# The 'Ubytes' folder is in the same directory as this script
ubyte_folder = './Ubytes'

# Transformations: Convert to Tensor and Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print(f"Loading data from local folder: {ubyte_folder}")

try:
    train_dataset = LocalFashionMNIST(root_dir=ubyte_folder, kind='train', transform=transform)
    test_dataset = LocalFashionMNIST(root_dir=ubyte_folder, kind='t10k', transform=transform)
except FileNotFoundError as e:
    print(f"\nERROR: Could not find files in {ubyte_folder}")
    print("Please ensure the 'Ubytes' folder is next to this .py file and contains:")
    print("- train-images-idx3-ubyte")
    print("- train-labels-idx1-ubyte")
    print("- t10k-images-idx3-ubyte")
    print("- t10k-labels-idx1-ubyte")
    exit()

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 2. CNN Architecture Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 14x14

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 7x7

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 Output Classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)

# 3. Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}')

print("Training finished.")

# 4. Export to ONNX
# Set model to evaluation mode
model.eval()
# Create a dummy input to trace the graph
dummy_input = torch.randn(1, 1, 28, 28).to(device)
onnx_path = "fashion_mnist_cnn.onnx"

print(f"Exporting model to {onnx_path}...")
torch.onnx.export(model,  # Model to export
                  dummy_input,  # Dummy input
                  onnx_path,  # Output file name
                  export_params=True,  # Save trained weights
                  opset_version=11,  # ONNX version
                  do_constant_folding=False,
                  input_names=['input'],  # Input node name
                  output_names=['output'],  # Output node name
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("Export completed successfully.")
