# Airbnb Vienna Price Prediction (NN from Scratch)

This project implements a Custom Neural Network (ANN) from scratch using NumPy and Pandas to predict Airbnb listing prices in Vienna. It covers the entire machine learning pipeline, including data extraction, one-hot encoding, and manual backpropagation.

## Data Source and Engineering

The dataset is sourced from the official **Airbnb** database (Inside Airbnb), specifically focusing on the listings in Vienna. This real-world dataset provides a comprehensive look at urban housing market variables.



The preprocessing involves the following stages:

1. **Data Acquisition:** Raw listing data containing geolocation, pricing, and room characteristics.
2. **Feature Engineering:** Selection of 8 critical features including latitude, longitude, room type, and availability.
3. **Categorical Encoding:** Transformation of the `room_type` variable using One-Hot Encoding for compatibility with the neural network.
4. **Handling Missing Values:** Median imputation for numerical columns to maintain distribution integrity.
5. **Min-Max Normalization:** Features and the target variable (price) are scaled to the $[0, 1]$ range for faster convergence:
   $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

## Key Features

* **Manual Framework Implementation:** No Keras, PyTorch, or TensorFlow. The backpropagation and gradient descent logic are implemented from the ground up.
* **End-to-End Pipeline:** Includes a dedicated data processing module for cleaning, encoding, and normalizing the Airbnb dataset.
* **Vectorized Implementation:** Fully optimized matrix operations using NumPy for efficient training performance.
* **Mathematical Transparency:** Uses ReLU activation and MSE loss, visualized through training loss curves.

## Neural Network Architecture

The model is a multi-layer perceptron (MLP) with the following technical specifications:

* **Input Layer:** Dimension corresponds to the feature count after one-hot encoding.
* **Hidden Layer:** 16 neurons with Rectified Linear Unit (ReLU) activation.
  * ReLU Formula: $A = \max(0, Z)$
* **Output Layer:** 1 neuron for linear regression to predict the continuous price value.
* **Optimization:** Gradient Descent with manual calculation of derivatives.



### Backpropagation Logic
Gradients are calculated as follows:
$$\frac{\partial Loss}{\partial W_2} = A_1^T \cdot dZ_2$$
$$\frac{\partial Loss}{\partial W_1} = X^T \cdot dZ_1$$



## Installation and Usage

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/mertiodas/vienna-airbnb-price-prediction.git](https://github.com/mertiodas/vienna-airbnb-price-prediction.git)
   cd vienna-airbnb-price-prediction
2. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
4. **Execute the Project:**
   ```bash
   python main.py

**Results**
The model generates a loss history graph upon completion. A successful training session is indicated by a steady decrease in Mean Squared Error (MSE), demonstrating that the weights are successfully optimizing for the Vienna housing market data.

**Author**
Mert - Electrical & Electronics Engineering Student
