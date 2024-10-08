import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
import time
import logging
from collections import Counter
import math
from datetime import datetime
import random

# Characteristics for logging
num_clients = 20
selection_rate = 0.2
alpha = 0.1  # Dirichlet distribution parameter for non-IID
iid_type = "non-iid" if alpha < 1 else "iid"  # Characterize dataset
algorithm = "fedavg"
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Log file name format: includes date, algorithm, non-IID/IID type, and number of clients
log_file_name = f"federated_learning_{algorithm}_{iid_type}_{num_clients}clients_{date_str}.txt"

# Ensure the directory for logs exists
log_directory = './experiment_logs'  # You can change this path as needed
os.makedirs(log_directory, exist_ok=True)  # Create directory if it doesn't exist
log_file_path = os.path.join(log_directory, log_file_name)

# Set up logging with dynamic file name using full log_file_path
logging.basicConfig(filename=log_file_path, level=logging.INFO)

# Client failure parameters
failure_probability = 0.2  # 20% chance of failure per round
rejoin_probability = 0.3   # 30% chance that a failed client will rejoin in the next round

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Download and load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader for the test dataset
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def get_model_size(model):
    """Calculates the size of the model in bytes."""
    total_size = 0
    for param in model.parameters():
        total_size += param.element_size() * param.nelement()  # Size of each element * number of elements
    return total_size

# Function to split dataset using Dirichlet distribution for non-IID split
def split_dataset_by_dirichlet(dataset, num_clients, alpha):
    indices = np.arange(len(dataset))
    label_array = np.array([dataset[i][1] for i in indices])  # Get labels from the dataset
    client_indices = [[] for _ in range(num_clients)]

    # Split data for each class based on Dirichlet distribution
    for label in np.unique(label_array):
        class_indices = indices[label_array == label]
        np.random.shuffle(class_indices)
        class_split = np.random.dirichlet([alpha] * num_clients) * len(class_indices)
        class_split = np.round(class_split).astype(int)
        class_split = np.cumsum(class_split).astype(int)

        start = 0
        for client_id in range(num_clients):
            client_indices[client_id].extend(class_indices[start:class_split[client_id]])
            start = class_split[client_id]

    # Return indices for each client
    return client_indices

# Function to calculate entropy given label counts
def calculate_entropy(label_counts):
    total_samples = sum(label_counts.values())
    entropy = 0.0
    for count in label_counts.values():
        if count > 0:
            p_i = count / total_samples
            entropy -= p_i * math.log2(p_i)
    return entropy

# Non-IID data partition
client_data_indices = split_dataset_by_dirichlet(train_dataset, num_clients, alpha)

# Create a DataLoader for each client
client_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=64, shuffle=True) 
                  for indices in client_data_indices]

# Calculate,log and store entropy for each client
client_entropies = []
for i, indices in enumerate(client_data_indices):
    # Extract labels for the current client
    labels = [train_dataset[idx][1] for idx in indices]
    # Count occurrences of each label
    label_counts = Counter(labels)
    # Calculate entropy
    entropy = calculate_entropy(label_counts)
    client_entropies.append((i, entropy))  # Store (client_id, entropy) tuples
    # Log the entropy
    logging.info(f"Client {i+1} Label Entropy: {entropy:.4f}")
    print(f"Client {i+1} Label Entropy: {entropy:.4f}")

# Print number of samples per client (for validation)
for i, loader in enumerate(client_loaders):
    print(f"Client {i+1}: {len(loader.dataset)} samples")

# Rank clients by entropy in descending order
client_entropies.sort(key=lambda x: x[1], reverse=True)

#Select the top 20% of clients by entropy for training
num_selected_clients = max(1, int(selection_rate * len(client_entropies)))  # Ensure at least one client is selected
selected_clients = [client_id for client_id, _ in client_entropies[:num_selected_clients]]

logging.info(f"Selected top {num_selected_clients} clients for training based on label entropy.")
print(f"Selected top {num_selected_clients} clients for training: {selected_clients}")

# Define the CNN architecture for CIFAR-10 (reduced complexity for optimization)
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
learning_rate = 0.001
num_rounds = 30  # Reduced number of rounds for optimization
num_local_epochs = 2  # Reduced number of local epochs

# FedAvg Algorithm to average the model weights
def fed_avg(global_model, client_models):
    global_weights = global_model.state_dict()
    for key in global_weights.keys():
        global_weights[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))], dim=0).mean(dim=0)
    global_model.load_state_dict(global_weights)
    return global_model

# Train each client locally
def train_client(client_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in client_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(client_loader), accuracy

# Evaluate the model
def evaluate_model(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    return accuracy

# Track active clients (start with all clients active)
client_active_status = [True] * num_clients  # Initially, all clients are active

# Main federated learning loop
global_model = CIFAR10_CNN().to(device)
criterion = nn.CrossEntropyLoss()

# Initialize log
logging.info(f"Starting federated learning with {num_clients} clients")

for round in range(num_rounds):
    start_time = time.time()
    logging.info(f"Communication Round {round + 1}")
    
    client_models = []
    round_loss = []
    round_accuracy = []

    # Local training on each client
    for client_id in selected_clients:
        # Simulate client failure
        if client_active_status[client_id]:  # Client is active
            if random.random() < failure_probability:
                client_active_status[client_id] = False  # Client fails (quits)
                logging.info(f"Client {client_id + 1} failed and will not participate in this round.")
                continue  # Skip the training for this client
            elif not client_active_status[client_id] and random.random() < rejoin_probability:  # Client is currently inactive (failed previously)
                client_active_status[client_id] = True  # Client rejoins
                logging.info(f"Client {client_id + 1} rejoins the training in this round.")

        # If the client is active, proceed with training
        if client_active_status[client_id]:
            local_model = CIFAR10_CNN().to(device)
            local_model.load_state_dict(global_model.state_dict())  # Start from global model
            optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

            # Training each client
            client_loss, client_accuracy = train_client(client_loaders[client_id], local_model, criterion, optimizer, device)
            round_loss.append(client_loss)
            round_accuracy.append(client_accuracy)

            # Calculate size of the local model
            local_model_size = get_model_size(local_model)
            logging.info(f"Client {client_id} uploaded model size: {local_model_size / (1024 * 1024):.2f} MB")

            client_models.append(local_model)
            logging.info(f"Client {client_id + 1} - Loss: {client_loss:.4f}, Accuracy: {client_accuracy:.2f}%")

    # Aggregate client models into the global model
    if client_models:
        global_model = fed_avg(global_model, client_models)
        # Calculate size of the global model
        global_model_size = get_model_size(global_model)
        logging.info(f"Global model size after this round: {global_model_size / (1024 * 1024):.2f} MB")
    else:
        logging.info("No clients participated in this round.")

    # Evaluate global model on the test set
    test_accuracy = evaluate_model(test_loader, global_model, device)
    logging.info(f"Test Accuracy after round {round + 1}: {test_accuracy:.2f}%")

    elapsed_time = time.time() - start_time
    logging.info(f"Round {round + 1} completed in {elapsed_time:.2f} seconds\n")

logging.info("Federated learning process finished successfully.")

