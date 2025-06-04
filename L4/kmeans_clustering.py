import numpy as np
from sklearn.cluster import KMeans

def perform_kmeans(X, n_clusters, n_init=10, max_iter=300, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                    n_init=n_init, max_iter=max_iter, 
                    random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    return kmeans, labels

def get_cluster_digit_mapping(labels, true_labels, n_clusters):
    digit_counts = np.zeros((n_clusters, 10))
    
    for cluster in range(n_clusters):
        mask = (labels == cluster)
        digits_in_cluster = true_labels[mask]
        
        for digit in range(10):
            digit_counts[cluster, digit] = np.sum(digits_in_cluster == digit)
    
    cluster_to_digit = np.argmax(digit_counts, axis=1)
    
    return cluster_to_digit, digit_counts