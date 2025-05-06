import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_neural_network(model, train_loader, device, epochs=10, lr=0.001):
    print("Rozpoczynamy trening sieci neuronowej...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  
            outputs = model(images)  
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 

            running_loss += loss.item()
            if batch_idx % 100 == 99: 
                print(f"[{epoch+1}, {batch_idx+1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Zakończono trening sieci neuronowej.")

def train_random_forest(train_loader):
    print("Rozpoczynamy trening klasyfikatora Random Forest...")
    X_train = []
    y_train = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(images.size(0), -1).numpy()
        X_train.extend(images)
        y_train.extend(labels.numpy())

        if batch_idx % 100 == 99:  
            print(f"Załadowano {batch_idx+1} batchy danych do Random Forest.")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)

    print("Zakończono trening klasyfikatora Random Forest.")
    return rf_model
