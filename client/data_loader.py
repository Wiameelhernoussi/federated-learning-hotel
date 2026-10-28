import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_partition(client_id, partition_type="iid", seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 5 000 exemples par client (équilibre vitesse/représentativité en environnement Docker CPU)
    num_samples = 5000
    
    if partition_type == "iid":
        # Partition aléatoire uniforme
        indices = np.random.permutation(len(trainset))
        start_idx = client_id * num_samples
        client_indices = indices[start_idx : start_idx + num_samples]
    else:
        # Partition Non-IID : classes majoritaires
        targets = np.array(trainset.targets)
        if client_id == 0:
            majority_classes = [0, 1]
        elif client_id == 1:
            majority_classes = [4, 5, 6]
        else:
            majority_classes = [8, 9]
            
        majority_indices = np.where(np.isin(targets, majority_classes))[0]
        other_indices = np.where(~np.isin(targets, majority_classes))[0]
        
        np.random.shuffle(majority_indices)
        np.random.shuffle(other_indices)
        
        # 80% de données majoritaires, 20% autres pour que ce soit non-IID mais apprenable
        num_majority = int(num_samples * 0.8)
        num_other = num_samples - num_majority
        
        # Prendre autant que possible dans les classes majoritaires
        client_indices = np.concatenate([
            majority_indices[:num_majority],
            other_indices[:num_other]
        ])
        # Si on n'a pas assez d'éléments majoritaires, on complète
        if len(client_indices) < num_samples:
            remaining = num_samples - len(client_indices)
            client_indices = np.concatenate([client_indices, other_indices[num_other:num_other+remaining]])

    train_subset = Subset(trainset, client_indices)
    
    # Validation data: tout le testset pour simuler l'évaluation globale sur chaque client
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    
    return trainloader, testloader, len(train_subset)
