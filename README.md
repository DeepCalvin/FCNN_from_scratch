# FCNN from Scratch using NumPy

This project implements a **Fully Connected Neural Network (FCNN)** from scratch using **pure NumPy** – no frameworks like TensorFlow or PyTorch involved! It's built for binary classification tasks and trained on the classic **Iris dataset**.

## Features

- Dense Layer with custom weight initialization
- Activation Functions: **ReLU**, **Sigmoid**
- Forward & Backward Propagation
- Mean Squared Error (MSE) Loss
- Mini-batch Gradient Descent
- Custom training loop
- Sample task: Classify *Setosa vs. Not Setosa* from the Iris dataset

## Architecture

The architecture used in this example:
Input (4 features)
-> Dense(128, ReLU)
-> Dense(128, ReLU)
-> Dense(1, Sigmoid)
