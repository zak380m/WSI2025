from data_loader import load_emnist_mnist
from dbscan_clustering import (find_optimal_eps, perform_dbscan, 
                              get_cluster_digit_mapping_dbscan,
                              evaluate_dbscan_performance)
from visualization import (plot_cluster_distribution_dbscan, 
                         plot_dbscan_results)
import numpy as np
import umap

def run_task2():
    X_sample, y_sample, _, _ = load_emnist_mnist()
    X_sample = X_sample[:70000]
    y_sample = y_sample[:70000]
    
    print("Reducing dimensionality with UMAP...")
    n_components = 15 
    
    reducer = umap.UMAP(n_components=n_components, 
                        random_state=42,
                        n_neighbors=10,
                        min_dist=0.1) 
    
    X_reduced = reducer.fit_transform(X_sample)
    
    print(f"Reduced dimensions from {X_sample.shape[1]} to {X_reduced.shape[1]}")
    
    print("Generating k-distance graph for epsilon estimation...")
    suggested_eps = find_optimal_eps(X_reduced)
    print(f"Suggested eps value: {suggested_eps:.2f}")
    
    eps_values = [0.27, 0.30, 0.33, 0.36]
    min_samples_values = [25, 30, 35]
    
    best_params = None
    best_score = -np.inf
    best_results = None
    
    print("\nStarting grid search...")
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\nTrying eps={eps:.2f}, min_samples={min_samples}")
            
            _, labels, n_clusters = perform_dbscan(X_reduced, eps, min_samples)
            
            cluster_to_digit, digit_counts, _ = get_cluster_digit_mapping_dbscan(labels, y_sample)
            accuracy, noise_percentage, misclassification = evaluate_dbscan_performance(
                labels, y_sample, cluster_to_digit)
            
            score = accuracy * (1 - noise_percentage/100)
            
            print(f"Clusters: {n_clusters}, Noise: {noise_percentage:.1f}%, "
                  f"Accuracy: {accuracy:.1f}%, Misclassification: {misclassification:.1f}%")
            
            if score > best_score and n_clusters <= 30:
                best_score = score
                best_params = (eps, min_samples)
                best_results = {
                    'labels': labels,
                    'cluster_to_digit': cluster_to_digit,
                    'digit_counts': digit_counts,
                    'n_clusters': n_clusters,
                    'accuracy': accuracy,
                    'noise_percentage': noise_percentage,
                    'misclassification': misclassification
                }
    
    if best_results:
        print("\n=== Best Results ===")
        print(f"Parameters: eps={best_params[0]:.2f}, min_samples={best_params[1]}")
        print(f"Clusters: {best_results['n_clusters']}")
        print(f"Noise percentage: {best_results['noise_percentage']:.1f}%")
        print(f"Accuracy: {best_results['accuracy']:.1f}%")
        print(f"Misclassification: {best_results['misclassification']:.1f}%")
        
        plot_dbscan_results(X_reduced, best_results['labels'], best_results['n_clusters'])
        plot_cluster_distribution_dbscan(best_results['digit_counts'], best_results['n_clusters'])
        
        from visualization import plot_example_digits_per_cluster
        plot_example_digits_per_cluster(X_sample, best_results['labels'], 
                                      best_results['n_clusters'], 
                                      best_results['cluster_to_digit'])
    else:
        print("No suitable parameters found with <= 30 clusters")

if __name__ == "__main__":
    run_task2()