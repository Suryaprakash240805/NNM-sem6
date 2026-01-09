import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=3, hidden1_size=4, hidden2_size=3, output_size=1):
        # Initialize weights and biases
        # W1: (input_size, hidden1_size)
        self.W1 = np.random.randn(input_size, hidden1_size)
        self.b1 = np.zeros((1, hidden1_size))
        
        # W2: (hidden1_size, hidden2_size)
        self.W2 = np.random.randn(hidden1_size, hidden2_size)
        self.b2 = np.zeros((1, hidden2_size))
        
        # W3: (hidden2_size, output_size)
        self.W3 = np.random.randn(hidden2_size, output_size)
        self.b3 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # x is the output of the sigmoid function
        return x * (1 - x)

    def forward(self, X):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        # Output Layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3

    def compute_loss(self, y_true, y_pred):
        # Mean Squared Error
        return np.mean(np.square(y_true - y_pred))

    def backward(self, X, y_true, learning_rate):
        # Number of samples
        m = X.shape[0]
        
        # Output error
        dz3 = self.a3 - y_true # dLoss/dz3 (for MSE + Sigmoid)
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden Layer 2 error
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.sigmoid_derivative(self.a2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden Layer 1 error
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

if __name__ == "__main__":
    # Create network
    nn = NeuralNetwork(input_size=3, hidden1_size=4, hidden2_size=3, output_size=1)
    
    # Dummy data: 3 inputs, 1 output (XOR style or random)
    X = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,1]])
    y = np.array([[0], [1], [1], [0]]) # Just some targets
    
    print("Training Started...")
    for epoch in range(100):
        # Forward pass
        output = nn.forward(X)
        
        # Compute loss
        loss = nn.compute_loss(y, output)
        
        # Backward pass
        nn.backward(X, y, learning_rate=0.1)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    print("\nFinal Predictions:")
    print(nn.forward(X))
    print("\nActual Targets:")
    print(y)
