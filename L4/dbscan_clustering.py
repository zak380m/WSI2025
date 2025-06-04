import numpy as np
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
    n_samples = X.shape[0]
    
    labels = np.full(n_samples, -1)
    
    neighbors = []
    for i in range(n_samples):
        distances = np.sqrt(np.sum((X - X[i])**2, axis=1))
        neighbors.append(np.where(distances <= eps)[0])
    
    core_samples = np.array([i for i in range(n_samples) if len(neighbors[i]) >= min_samples])
    
    cluster_id = 0
    
    for point_idx in core_samples:
        if labels[point_idx] != -1:
            continue
            
        labels[point_idx] = cluster_id
        
        seeds = list(neighbors[point_idx])
        
        i = 0
        while i < len(seeds):
            current_point = seeds[i]
            
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
                
                if current_point in core_samples:
                    new_neighbors = neighbors[current_point]
                    for neighbor in new_neighbors:
                        if neighbor not in seeds:
                            seeds.append(neighbor)
            
            i += 1
        
        cluster_id += 1
        
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    return labels, n_clusters

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