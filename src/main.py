import pandas as pd
from neuralnetwork import NeuralNetwork
from process import split
import numpy as np
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = split("../data/extracted.csv")
x_train = x_train.astype(np.float64)
x_test  = x_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test  = y_test.astype(np.float64)


def main():

    nn = NeuralNetwork(input_size=x_train.shape[1], hidden_size=16, output_size=1, learning_rate=0.01, epochs=100)
    nn.train(x_train, y_train)
    predictions = nn.predict(x_test)
    plt.figure(figsize=(8, 5))
    plt.plot(nn.loss_history, color='blue')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.show()

main()