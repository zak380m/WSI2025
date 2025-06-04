import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cluster_distribution(digit_counts, n_clusters):
    percentages = digit_counts / digit_counts.sum(axis=1, keepdims=True) * 100
    
    plt.figure(figsize=(12, 8))
    plt.imshow(percentages, cmap='Blues', aspect='auto')
    plt.colorbar(label='Percentage (%)')
    plt.xlabel('Digit')
    plt.ylabel('Cluster')
    plt.title(f'Percentage Distribution of Digits in {n_clusters} Clusters')
    plt.xticks(range(10))
    plt.yticks(range(n_clusters))
    
    for i in range(n_clusters):
        for j in range(10):
            plt.text(j, i, f"{percentages[i, j]:.1f}%", 
                    ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.show()

def plot_centroids(centroids, n_clusters):
    centroids_images = centroids.reshape(-1, 28, 28)
    
    rows = int(np.ceil(np.sqrt(n_clusters)))
    cols = int(np.ceil(n_clusters / rows))
    
    plt.figure(figsize=(2*cols, 2*rows))
    for i in range(n_clusters):
        plt.subplot(rows, cols, i+1)
        plt.imshow(centroids_images[i], cmap='gray')
        plt.title(f'Cluster {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def plot_cluster_distribution_dbscan(digit_counts, n_clusters):
    percentages = digit_counts / digit_counts.sum(axis=1, keepdims=True) * 100
    
    plt.figure(figsize=(12, 8))
    plt.imshow(percentages, cmap='Blues', aspect='auto')
    plt.colorbar(label='Percentage (%)')
    plt.xlabel('Digit')
    plt.ylabel('Cluster (last is noise)')
    plt.title(f'Percentage Distribution of Digits in {n_clusters} Clusters (DBSCAN)')
    plt.xticks(range(10))
    plt.yticks(range(n_clusters + 1))  
    
    for i in range(n_clusters + 1):
        for j in range(10):
            if digit_counts[i, j] > 0: 
                plt.text(j, i, f"{percentages[i, j]:.1f}%", 
                        ha="center", va="center", color="black")
    
    plt.tight_layout()
    filepath = os.path.join('plots', 'dbscan_clusters.png')
    plt.savefig(filepath)
    plt.close()

def plot_dbscan_results(X, labels, n_clusters):
    from sklearn.manifold import TSNE
    
    if len(X) > 5000:
        indices = np.random.choice(len(X), 5000, replace=False)
        X_vis = X[indices]
        labels_vis = labels[indices]
    else:
        X_vis = X
        labels_vis = labels
    
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X_vis)
    
    plt.figure(figsize=(10, 8))
    
    noise_mask = (labels_vis == -1)
    if np.any(noise_mask):
        plt.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], 
                    c='gray', alpha=0.5, label='Noise', s=10)
    
    for cluster  in range(n_clusters):
        cluster_mask = (labels_vis == cluster)
        plt.scatter(X_2d[cluster_mask, 0], X_2d[cluster_mask, 1], 
                    alpha=0.7, label=f'Cluster {cluster}', s=15)
    
    plt.title(f'DBSCAN Clustering Results ({n_clusters} clusters)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filepath = os.path.join('plots', 'dbscan_clusters_matrix.png')
    plt.savefig(filepath)
    plt.close()
    
def plot_k_distance_graph(distances):
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel('k-NN distance')
    plt.title('K-distance Graph for Epsilon Estimation')
    
    plt.tight_layout()
    filepath = os.path.join('plots', 'eps.png')
    plt.savefig(filepath)
    plt.close()
    
def plot_example_digits_per_cluster(X_original, labels, n_clusters, cluster_to_digit, max_examples=5):
    plt.figure(figsize=(15, 3 * (n_clusters + 1)))
    
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        sample_indices = np.random.choice(
            cluster_indices, 
            min(max_examples, len(cluster_indices)), 
            replace=False
        )
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(n_clusters + 1, max_examples, cluster * max_examples + i + 1)
            plt.imshow(X_original[idx].reshape(28, 28), cmap='gray')
            plt.title(f"Cluster {cluster}\nPredicted: {cluster_to_digit[cluster]}")
            plt.axis('off')
    
    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) > 0:
        sample_indices = np.random.choice(
            noise_indices, 
            min(max_examples, len(noise_indices)), 
            replace=False
        )
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(n_clusters + 1, max_examples, n_clusters * max_examples + i + 1)
            plt.imshow(X_original[idx].reshape(28, 28), cmap='gray')
            plt.title(f"Noise (-1)")
            plt.axis('off')
    
    plt.tight_layout()
    filepath = os.path.join('plots', 'dbscan_clusters_images.png')
    plt.savefig(filepath)
    plt.close()