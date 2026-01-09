import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Hyperparameters
input_size = 3
hidden1_size = 4
hidden2_size = 3
output_size = 1
learning_rate = 0.1
epochs = 1000

W1 = np.random.randn(input_size, hidden1_size)
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size)
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size)
b3 = np.zeros((1, output_size))

X = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])

print("Procedural Training Started...")

for epoch in range(epochs):
    # Layer 1
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    
    # Layer 2
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Output Layer
    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)
    
    # Loss calculation
    loss = np.mean(np.square(y - a3))
    
    # --- Backward Pass (Gradients) ---
    m = X.shape[0]
    
    # Output Layer Gradients
    dz3 = a3 - y
    dW3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m
    
    # Hidden Layer 2 Gradients
    da2 = np.dot(dz3, W3.T)
    dz2 = da2 * sigmoid_derivative(a2)
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    
    # Hidden Layer 1 Gradients
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal Predictions:")
print(a3)
print("\nActual Targets:")
print(y)
