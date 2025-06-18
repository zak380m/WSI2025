import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class NeuralNetwork:
    def __init__(self, activation='sigmoid', learning_rate=0.1):
        self.weights1 = np.random.randn(2, 4)  # input to hidden
        self.weights2 = np.random.randn(4, 1)  # hidden to output
        self.bias1 = np.zeros((1, 4))
        self.bias2 = np.zeros((1, 1))
        self.activation = activation
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        # Hidden layer
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        if self.activation == 'sigmoid':
            self.hidden_output = self.sigmoid(self.hidden_input)
        else:  # ReLU
            self.hidden_output = self.relu(self.hidden_input)
            
        # Output layer
        self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        self.output = self.sigmoid(self.output_input)
        return self.output
    
    def backward(self, X, y, output):
        # Output layer error
        error = y - output
        output_delta = error * self.sigmoid_derivative(output)
        
        # Hidden layer error
        hidden_error = output_delta.dot(self.weights2.T)
        if self.activation == 'sigmoid':
            hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        else:  # ReLU
            hidden_delta = hidden_error * self.relu_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights2 += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights1 += X.T.dot(hidden_delta) * self.learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs=10000):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean(np.square(y - output))
            losses.append(loss)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        return losses
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

def normalize_l1(X):
    return X / np.sum(np.abs(X), axis=1, keepdims=True)

def normalize_l2(X):
    return X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))

# Generate training data
np.random.seed(42)
X_train = np.random.uniform(-1, 1, (1000, 2))
X_train = X_train[(X_train[:, 0] != 0) & (X_train[:, 1] != 0)]  # remove zeros
y_train = ((X_train[:, 0] * X_train[:, 1]) > 0).astype(int).reshape(-1, 1)

# Create different normalization versions
X_train_l1 = normalize_l1(X_train)
X_train_l2 = normalize_l2(X_train)

# Test different configurations
configurations = [
    {'name': 'Unnormalized σ', 'data': X_train, 'activation': 'sigmoid'},
    {'name': 'L1 Normalized σ', 'data': X_train_l1, 'activation': 'sigmoid'},
    {'name': 'L2 Normalized σ', 'data': X_train_l2, 'activation': 'sigmoid'},
    {'name': 'Unnormalized ReLU', 'data': X_train, 'activation': 'relu'},
    {'name': 'L1 Normalized ReLU', 'data': X_train_l1, 'activation': 'relu'},
    {'name': 'L2 Normalized ReLU', 'data': X_train_l2, 'activation': 'relu'},
]

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
results = []

for config in configurations:
    for lr in learning_rates:
        print(f"\nTraining {config['name']} with learning rate {lr}")
        nn = NeuralNetwork(activation=config['activation'], learning_rate=lr)
        losses = nn.train(config['data'], y_train, epochs=5000)
        
        # Test accuracy
        X_test = np.random.uniform(-1, 1, (100, 2))
        X_test = X_test[(X_test[:, 0] != 0) & (X_test[:, 1] != 0)]
        y_test = ((X_test[:, 0] * X_test[:, 1]) > 0).astype(int).reshape(-1, 1)
        
        if config['name'].startswith('L1'):
            X_test_norm = normalize_l1(X_test)
        elif config['name'].startswith('L2'):
            X_test_norm = normalize_l2(X_test)
        else:
            X_test_norm = X_test
            
        predictions = nn.predict(X_test_norm)
        accuracy = np.mean(predictions == y_test)
        
        results.append({
            'config': config['name'],
            'learning_rate': lr,
            'final_loss': losses[-1],
            'accuracy': accuracy,
            'loss_curve': losses
        })

# Plot results
plt.figure(figsize=(15, 10))
for i, config in enumerate(configurations):
    for lr in learning_rates:
        result = next(r for r in results if r['config'] == config['name'] and r['learning_rate'] == lr)
        plt.subplot(2, 3, i+1)
        plt.plot(result['loss_curve'], label=f'LR={lr}')
    plt.title(config['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
plt.tight_layout()
plt.show()

# Print summary table
print("\nSummary of Results:")
print("Configuration".ljust(25), "Learning Rate".ljust(15), "Final Loss".ljust(15), "Accuracy")
for result in results:
    print(result['config'].ljust(25), 
          str(result['learning_rate']).ljust(15),
          f"{result['final_loss']:.6f}".ljust(15),
          f"{result['accuracy']:.2%}")
    
# Save results to CSV
df = pd.DataFrame(results)
df = df[['config', 'learning_rate', 'final_loss', 'accuracy']]
df.columns = ['Configuration', 'Learning_Rate', 'Final_Loss', 'Accuracy']
df.to_csv('neural_network_results.csv', index=False)
print(f"\nResults saved to neural_network_results.csv")
    
# Visualization of all three normalizations
plt.figure(figsize=(15, 5))

# Generate sample data
np.random.seed(42)
X = np.random.uniform(-1, 1, (200, 2))
X = X[(X[:, 0] != 0) & (X[:, 1] != 0)][:100]  # Remove zeros

X_l1 = normalize_l1(X)
X_l2 = normalize_l2(X)

# Plot 1: Unnormalized (Square)
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c='blue')
plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'r-', linewidth=2, label='Boundary')
plt.title('Unnormalized Data\n(Square)')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.grid(True)
plt.axis('equal')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)

# Plot 2: L1 Normalized (Diamond)
plt.subplot(1, 3, 2)
plt.scatter(X_l1[:, 0], X_l1[:, 1], alpha=0.6, c='green')
plt.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], 'r-', linewidth=2, label='L1 unit ball')
plt.title('L1 Normalized\n(Diamond)')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.grid(True)
plt.axis('equal')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)

# Plot 3: L2 Normalized (Circle)
plt.subplot(1, 3, 3)
plt.scatter(X_l2[:, 0], X_l2[:, 1], alpha=0.6, c='orange')
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, label='L2 unit ball')
plt.title('L2 Normalized\n(Circle)')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.grid(True)
plt.axis('equal')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)

plt.tight_layout()
plt.show()