
import torch
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torch import nn, optim

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for models like ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load SVHN dataset
train_data = SVHN(root='./data', split='train', transform=transform, download=True)
test_data = SVHN(root='./data', split='test', transform=transform, download=True)

# Using a subset (25%) of the dataset for training and testing
subset_indices_train = torch.randperm(len(train_data))[:len(train_data)//4]
subset_indices_test = torch.randperm(len(test_data))[:len(test_data)//4]

train_data_subset = Subset(train_data, subset_indices_train)
test_data_subset = Subset(test_data, subset_indices_test)

train_loader = DataLoader(train_data_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data_subset, batch_size=64, shuffle=False)

# Function to load a pretrained model
def load_pretrained_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    # Add other models as needed
    model.fc = nn.Linear(model.fc.in_features, 10)  # Assuming 10 classes for SVHN
    return model

# Training and evaluation logic here...

if __name__ == '__main__':
    model_names = ['resnet18']  # Extend this list with other models as needed
    results = {}
    
    for model_name in model_names:
        model = load_pretrained_model(model_name)
        # Train and evaluate the model...
        results[model_name] = {'accuracy': None}  # Populate with actual accuracy

    print(results)
