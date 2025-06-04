from data_loader import load_emnist_mnist
from kmeans_clustering import perform_kmeans, get_cluster_digit_mapping
from visualization import plot_cluster_distribution, plot_centroids

def run_task1():
    X_train, y_train, _, _ = load_emnist_mnist()
    
    cluster_counts = [10, 15, 20, 30]
    
    for n_clusters in cluster_counts:
        print(f"\n=== Running for {n_clusters} clusters ===")
        
        kmeans, labels = perform_kmeans(X_train, n_clusters)
        print(f"Inertia: {kmeans.inertia_:.2f}")
        
        cluster_to_digit, digit_counts = get_cluster_digit_mapping(labels, y_train, n_clusters)
        
        plot_cluster_distribution(digit_counts, n_clusters)
        
        plot_centroids(kmeans.cluster_centers_, n_clusters)
        
        print("Cluster to digit mapping:")
        for cluster, digit in enumerate(cluster_to_digit):
            print(f"Cluster {cluster}: Mostly digit {digit}")

if __name__ == "__main__":
    run_task1()