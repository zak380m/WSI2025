from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

CUSTOM_DATA_PATH = '/home/myszojele/WSI/L1/custom_digits'

transform = transforms.Compose([
    transforms.RandomRotation(degrees=(90, 90)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))  
])

transform_custom = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))   
])

def load_emnist_data(batch_size=64):
    train_dataset = datasets.EMNIST(root='./data', split='mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root='./data', split='mnist', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class CustomDigitsDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = int(os.path.basename(image_path)[0]) 
        if self.transform:
            image = self.transform(image)
        return image, label

def load_custom_data(batch_size=1):
    custom_dataset = CustomDigitsDataset(CUSTOM_DATA_PATH, transform=transform_custom)
    custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    return custom_loader
