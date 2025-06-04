import numpy as np
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor

def load_emnist_mnist():
    train_data = EMNIST(root='./data', split='byclass', train=True, download=True, transform=ToTensor())
    test_data = EMNIST(root='./data', split='byclass', train=False, download=True, transform=ToTensor())
    
    X_train = train_data.data.numpy()
    y_train = train_data.targets.numpy()
    X_test = test_data.data.numpy()
    y_test = test_data.targets.numpy()
    
    train_mask = y_train < 10
    test_mask = y_test < 10
    
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    X_train = X_train.reshape(-1, 28*28) / 255.0
    X_test = X_test.reshape(-1, 28*28) / 255.0
    
    X_train = X_train.reshape(-1, 28, 28)
    X_train = np.transpose(X_train, (0, 2, 1))   
    X_train = X_train.reshape(-1, 28*28)
    
    X_test = X_test.reshape(-1, 28, 28)
    X_test = np.transpose(X_test, (0, 2, 1))    
    X_test = X_test.reshape(-1, 28*28)
    
    return X_train, y_train, X_test, y_test