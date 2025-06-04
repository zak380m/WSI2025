import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def find_optimal_eps(X, k=5, show_plot=True):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    
    distances = np.sort(distances[:, -1])
    
    derivatives = np.diff(distances)
    elbow_index = np.argmax(derivatives > np.max(derivatives)*0.1)
    
    if show_plot:
        from visualization import plot_k_distance_graph
        plot_k_distance_graph(distances)
    
    return distances[elbow_index]

def perform_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    return dbscan, labels, n_clusters

def get_cluster_digit_mapping_dbscan(labels, true_labels):
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    digit_counts = np.zeros((n_clusters + 1, 10))
    
    for i, label in enumerate(labels):
        cluster = label if label != -1 else n_clusters
        digit = true_labels[i]
        digit_counts[cluster, digit] += 1
    
    cluster_to_digit = np.argmax(digit_counts[:-1], axis=1)
    
    return cluster_to_digit, digit_counts, n_clusters

def evaluate_dbscan_performance(labels, true_labels, cluster_to_digit):
    noise_mask = (labels == -1)
    noise_percentage = np.mean(noise_mask) * 100
    
    non_noise_mask = ~noise_mask
    non_noise_labels = labels[non_noise_mask]
    non_noise_true = true_labels[non_noise_mask]
    
    if len(non_noise_labels) > 0:
        predicted_digits = cluster_to_digit[non_noise_labels]
        correct = np.sum(predicted_digits == non_noise_true)
        accuracy = correct / len(non_noise_labels) * 100
        misclassification = 100 - accuracy
    else:
        accuracy = 0.0
        misclassification = 0.0
    
    return accuracy, noise_percentage, misclassification