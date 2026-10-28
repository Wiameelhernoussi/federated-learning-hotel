import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def get_optimizer(model, lr=0.01, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(model, trainloader, epochs, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    model.train()
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

def test(model, testloader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    return loss / len(testloader), accuracy
def train_fedprox(model, trainloader, global_weights, epochs, device="cpu", mu=0.01):
    """
    FedProx training: adds a proximal term (mu/2) * ||w - w_global||^2 to the loss
    to prevent client drift during non-IID training.
    global_weights: list of torch.Tensor (one per model parameter, no grad)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    model.train()

    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            # Standard cross-entropy loss
            ce_loss = criterion(outputs, labels)

            # Proximal term: penalise deviation from global model
            prox_loss = 0.0
            for w, w_g in zip(model.parameters(), global_weights):
                prox_loss += torch.norm(w - w_g.to(device)) ** 2

            total_loss = ce_loss + (mu / 2) * prox_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def get_global_weights_as_tensors(global_weight_arrays):
    """Convert a list of numpy arrays to a list of float32 tensors (no grad)."""
    return [torch.tensor(w, dtype=torch.float32) for w in global_weight_arrays]
