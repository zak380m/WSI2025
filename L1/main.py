import os
import torch
from train import train_neural_network, train_random_forest
from evaluate import test_model
from data_loader import load_emnist_data, load_custom_data
from model import NeuralNetwork
from model import ConvNet

NN_MODEL_PATH = 'nn_model.pth'
CNN_MODEL_PATH = 'cnn_model.pth'
RF_MODEL_PATH = 'rf_model.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Ładowanie danych EMNIST...")
train_loader, test_loader = load_emnist_data(batch_size=64)

print("Ładowanie własnych danych testowych (custom)...")
custom_loader = load_custom_data(batch_size=1)

nn_model = NeuralNetwork().to(device)
    
if os.path.exists(NN_MODEL_PATH):
    print(f"Wczytywanie wytrenowanego modelu sieci neuronowej z {NN_MODEL_PATH}...")
    nn_model.load_state_dict(torch.load(NN_MODEL_PATH))
else:
    print("Trenowanie sieci neuronowej...")
    train_neural_network(nn_model, train_loader, device, epochs=8)
    torch.save(nn_model.state_dict(), NN_MODEL_PATH)  
    print(f"Model sieci neuronowej zapisano do {NN_MODEL_PATH}.")
    
cnn_model = ConvNet().to(device)

if os.path.exists(CNN_MODEL_PATH):
    print(f"Wczytywanie wytrenowanego modelu konwolucyjnej sieci neuronowej z {CNN_MODEL_PATH}...")
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
else:
    print("Trenowanie konwolucyjnej sieci neuronowej (CNN)...")
    train_neural_network(cnn_model, train_loader, device, epochs=8)
    torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)
    print(f"Model konwolucyjnej sieci neuronowej zapisano do {CNN_MODEL_PATH}.")

if os.path.exists(RF_MODEL_PATH):
    print(f"Wczytywanie wytrenowanego modelu Random Forest z {RF_MODEL_PATH}...")
    rf_model = torch.load(RF_MODEL_PATH, weights_only=False)
else:
    print("Trenowanie Random Forest...")
    rf_model = train_random_forest(train_loader)
    torch.save(rf_model, RF_MODEL_PATH)
    print(f"Model Random Forest zapisano do {RF_MODEL_PATH}.")

print("\nTestowanie sieci neuronowej na danych EMNIST...")
test_model(nn_model, test_loader, device, is_neural_net=True)

print("\nTestowanie sieci neuronowej na własnych danych testowych...")
test_model(nn_model, custom_loader, device, is_neural_net=True)

print("\nTestowanie konwolucyjnej sieci neuronowej (CNN) na danych EMNIST...")
test_model(cnn_model, test_loader, device, is_neural_net=True)

print("\nTestowanie konwolucyjnej sieci neuronowej (CNN) na własnych danych testowych...")
test_model(cnn_model, custom_loader, device, is_neural_net=True)

print("\nTestowanie Random Forest na danych EMNIST...")
test_model(rf_model, test_loader, is_neural_net=False)

print("\nTestowanie Random Forest na własnych danych testowych...")
test_model(rf_model, custom_loader, is_neural_net=False)
