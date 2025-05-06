import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib import colors

# Funkcje pomocnicze do parsowania danych
def parse_loss_data(text):
    pattern = r'\[(\d+), (\d+)\] loss: (\d+\.\d+)'
    matches = re.findall(pattern, text)
    
    epochs = []
    batches = []
    losses = []
    
    for epoch, batch, loss in matches:
        epochs.append(int(epoch))
        batches.append(int(batch))
        losses.append(float(loss))
    
    return epochs, batches, losses

def parse_confusion_matrix(text, model_name, dataset_name):
    # Znajdź sekcję z wynikami testowania
    if model_name == 'Random Forest' and dataset_name == 'własnych danych testowych':
        # Specjalna obsługa dla RF na własnych danych
        section_pattern = r'Testowanie Random Forest na własnych danych testowych.*?Cyfra 0: \[(.*?)\].*?Cyfra 1: \[(.*?)\].*?Cyfra 2: \[(.*?)\].*?Cyfra 3: \[(.*?)\].*?Cyfra 4: \[(.*?)\].*?Cyfra 5: \[(.*?)\].*?Cyfra 6: \[(.*?)\].*?Cyfra 7: \[(.*?)\].*?Cyfra 8: \[(.*?)\].*?Cyfra 9: \[(.*?)\]'
    else:
        # Standardowa obsługa dla innych przypadków
        section_pattern = rf'Testowanie {model_name} na {dataset_name}.*?Cyfra 0: \[(.*?)\].*?Cyfra 1: \[(.*?)\].*?Cyfra 2: \[(.*?)\].*?Cyfra 3: \[(.*?)\].*?Cyfra 4: \[(.*?)\].*?Cyfra 5: \[(.*?)\].*?Cyfra 6: \[(.*?)\].*?Cyfra 7: \[(.*?)\].*?Cyfra 8: \[(.*?)\].*?Cyfra 9: \[(.*?)\]'
    
    matches = re.findall(section_pattern, text, re.DOTALL)
    
    if not matches:
        print(f"Nie znaleziono sekcji dla {model_name} i {dataset_name}")
        return None
    
    # Pierwsze dopasowanie (powinno być tylko jedno)
    rows = matches[0]
    
    if len(rows) != 10:
        print(f"Nieprawidłowa liczba wierszy w macierzy: {len(rows)}")
        return None
    
    confusion_matrix = []
    for row in rows:
        values = [int(x.strip()) for x in row.split()]
        confusion_matrix.append(values)
    
    return np.array(confusion_matrix)

def parse_metrics(text, model_name, dataset_name):
    pattern = rf'Testowanie {model_name} na {dataset_name}.*?Accuracy: (\d+\.\d+).*?Precision: (\d+\.\d+).*?Recall: (\d+\.\d+)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        accuracy, precision, recall = map(float, matches[0])
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall}
    else:
        print(f"Nie znaleziono metryk dla {model_name} i {dataset_name}")
        return None

# Wczytanie danych z pliku
with open('dane.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Parsowanie danych
nn_epochs, nn_batches, nn_losses = parse_loss_data(data.split('Trenowanie konwolucyjnej sieci neuronowej')[0])
cnn_epochs, cnn_batches, cnn_losses = parse_loss_data(data.split('Trenowanie konwolucyjnej sieci neuronowej')[1].split('Trenowanie Random Forest')[0])

# Metryki dla różnych modeli i zestawów danych
models = [
    ('sieci neuronowej', 'NN'),
    ('konwolucyjnej sieci neuronowej \(CNN\)', 'CNN'), 
    ('Random Forest', 'RF')
]
datasets = [
    ('danych EMNIST', 'EMNIST'),
    ('własnych danych testowych', 'Custom')
]

metrics = {}
for model_pattern, model_name in models:
    for dataset_pattern, dataset_name in datasets:
        key = f"{model_name} {dataset_name}"
        metrics[key] = parse_metrics(data, model_pattern, dataset_pattern)

# Macierze pomyłek
confusion_matrices = {}
for model_pattern, model_name in models:
    for dataset_pattern, dataset_name in datasets:
        key = f"{model_name} {dataset_name}"
        cm = parse_confusion_matrix(data, model_pattern, dataset_pattern)
        if cm is not None:
            confusion_matrices[key] = cm
            print(f"Macierz pomyłek dla {key}:\n{cm}\n")
        else:
            print(f"Nie udało się załadować macierzy pomyłek dla {key}")

# 1. Wykresy funkcji straty podczas treningu
plt.figure(figsize=(15, 6))

# Sieć neuronowa
plt.subplot(1, 2, 1)
plt.plot(range(len(nn_losses)), nn_losses, label='Funkcja straty')
plt.title('Funkcja straty - Sieć neuronowa')
plt.xlabel('Iteracje (batch)')
plt.ylabel('Strata')
plt.grid(True)

# CNN
plt.subplot(1, 2, 2)
plt.plot(range(len(cnn_losses)), cnn_losses, label='Funkcja straty', color='orange')
plt.title('Funkcja straty - CNN')
plt.xlabel('Iteracje (batch)')
plt.ylabel('Strata')
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Porównanie metryk dla różnych modeli na EMNIST
plt.figure(figsize=(10, 6))
x = np.arange(len(models))
width = 0.25

for i, (metric_name, color) in enumerate(zip(['accuracy', 'precision', 'recall'], ['blue', 'green', 'red'])):
    values = [metrics[f"{model_name} EMNIST"][metric_name] for _, model_name in models]
    plt.bar(x + i*width, values, width, label=metric_name.capitalize(), color=color)

plt.title('Porównanie metryk na danych EMNIST')
plt.xlabel('Model')
plt.ylabel('Wartość')
plt.xticks(x + width, [model_name for _, model_name in models])
plt.legend()
plt.grid(True, axis='y')
plt.ylim(0.9, 1.0)
plt.show()

# 3. Porównanie metryk dla różnych modeli na własnych danych
plt.figure(figsize=(10, 6))
x = np.arange(len(models))
width = 0.25

for i, (metric_name, color) in enumerate(zip(['accuracy', 'precision', 'recall'], ['blue', 'green', 'red'])):
    values = [metrics[f"{model_name} Custom"][metric_name] for _, model_name in models]
    plt.bar(x + i*width, values, width, label=metric_name.capitalize(), color=color)

plt.title('Porównanie metryk na własnych danych testowych')
plt.xlabel('Model')
plt.ylabel('Wartość')
plt.xticks(x + width, [model_name for _, model_name in models])
plt.legend()
plt.grid(True, axis='y')
plt.ylim(0, 1.0)
plt.show()

# 4. Macierze pomyłek dla modeli na danych EMNIST
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Macierze pomyłek na danych EMNIST', fontsize=16)

for i, (_, model_name) in enumerate(models):
    key = f"{model_name} EMNIST"
    if key in confusion_matrices:
        cm = confusion_matrices[key]
        
        # Normalizacja macierzy dla lepszej wizualizacji
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = axes[i].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        axes[i].set_title(model_name)
        axes[i].set_xlabel('Przewidziana etykieta')
        axes[i].set_ylabel('Prawdziwa etykieta')
        axes[i].set_xticks(np.arange(10))
        axes[i].set_yticks(np.arange(10))
        
        # Dodanie wartości w komórkach
        for j in range(10):
            for k in range(10):
                axes[i].text(k, j, f"{cm[j, k]}\n({cm_norm[j, k]:.1%}",
                            ha="center", va="center", color="black" if cm_norm[j, k] < 0.5 else "white",
                            fontsize=8)
        
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    else:
        axes[i].axis('off')
        axes[i].set_title(f"{model_name} - brak danych")

plt.tight_layout()
plt.show()

# 5. Macierze pomyłek dla modeli na własnych danych testowych
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Macierze pomyłek na własnych danych testowych', fontsize=16)

for i, (_, model_name) in enumerate(models):
    key = f"{model_name} Custom"
    if key in confusion_matrices:
        cm = confusion_matrices[key]
        
        # Dla własnych danych używamy bezwzględnych wartości
        im = axes[i].imshow(cm, cmap='Reds', vmin=0, vmax=3)
        axes[i].set_title(model_name)
        axes[i].set_xlabel('Przewidziana etykieta')
        axes[i].set_ylabel('Prawdziwa etykieta')
        axes[i].set_xticks(np.arange(10))
        axes[i].set_yticks(np.arange(10))
        
        # Dodanie wartości w komórkach
        for j in range(10):
            for k in range(10):
                axes[i].text(k, j, f"{cm[j, k]}",
                            ha="center", va="center", color="black" if cm[j, k] < 2 else "white",
                            fontsize=10)
        
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    else:
        axes[i].axis('off')
        axes[i].set_title(f"{model_name} - brak danych")

plt.tight_layout()
plt.show()

# 6. Porównanie dokładności między EMNIST a własnymi danymi
plt.figure(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

emnist_acc = [metrics[f"{model_name} EMNIST"]['accuracy'] for _, model_name in models]
custom_acc = [metrics[f"{model_name} Custom"]['accuracy'] for _, model_name in models]

plt.bar(x - width/2, emnist_acc, width, label='EMNIST', color='blue')
plt.bar(x + width/2, custom_acc, width, label='Własne dane', color='red')

plt.title('Porównanie dokładności między zbiorami danych')
plt.xlabel('Model')
plt.ylabel('Dokładność')
plt.xticks(x, [model_name for _, model_name in models])
plt.legend()
plt.grid(True, axis='y')
plt.show()