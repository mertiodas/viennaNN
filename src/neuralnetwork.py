import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        self.lr = learning_rate
        self.ep = epochs
        # 3 layers -> input, hidden, output. l-1 = weight & bias amount
        # starting weights from small numbers
        # biases from zeros
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.loss_history = []

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU ACTIVATION function!!!!
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2  # output. pretty linear

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true.reshape(-1, 1) - y_pred) ** 2)

    def backward(self, X, y, y_pred):
        m = X.shape[0]

        dZ2 = (y_pred - y.reshape(-1, 1)) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU derivative
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y):
        for i in range(self.ep):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred)
            self.loss_history.append(loss)
            print(f"Epoch {i}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)