import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def test_model(model, test_loader, device=None, is_neural_net=True):
    all_preds = []
    all_labels = []
    
    print("Rozpoczynamy testowanie...")

    with torch.no_grad():
        for images, labels in test_loader:
            if is_neural_net:
                images = images.to(device)
                model.eval()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
            else:
                images = images.view(images.size(0), -1).numpy()  
                predicted = model.predict(images)
                all_preds.extend(predicted)
            
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Wyniki testowania:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    for i in range(10):
        print(f"Cyfra {i}: {cm[i]}")

    return accuracy, precision, recall