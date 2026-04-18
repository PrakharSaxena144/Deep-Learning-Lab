# Deep-Learning-Lab

I will upload my college weekly assignments here related to Deep Learning Lab

week 1:

P1: 
Without using inbuilt functions: 1. Implement a perceptron from scratch for binary classification. Using the perceptron learning rule for training, classify linearly separable data (of your choice) and plot the decision boundary for the perceptron after training. 2. Implement XOR using network of perceptrons. Visualize the results.


week 2:

P2: 
Implement a multi-layered perceptron network with one output unit. 
1. Train it for different Boolean functions using the perceptron Learning Algorithm with thresholding logic (with no learning rate or error function).
2. Train it for different Boolean functions using the perceptron Learning Algorithm with sigmoid activation function (with no learning rate or error function).
   

week 3:

P3:  
Implement the Gradient Descent algorithm from first principles (without using automatic differentiation libraries) to minimize a multi-variable objective function (f(x,y)=x^2+y^2+10 sin⁡(x)+10 cos⁡(y) ). You must visualize the optimization trajectory on a 2D Contour Plot(error surface) to demonstrate how Learning Rate(at least 3 different rates) and Initial weight(at least 3 different ) value affect convergence.


week 4:

P4: Design and implement a feedforward neural network with exactly one hidden layer from scratch using only NumPy. The network should have 2 input neurons, 48 hidden neurons, and 1 output neuron with sigmoid activation. Use the make_moons dataset from scikit-learn (n_samples=1500, noise=0.15, random_state=42) for training and a separate test set of 500 samples (random_state=43). Train the model using mini-batch gradient descent batch size 64, learning rate 0.01, for 10,000 iterations. Use Binary Cross Entropy loss. Train four identical networks (same architecture, optimizer, hyperparameters, and random seed), changing only the hidden-layer activation function: (1) sigmoid, (2) tanh, (3) ReLU, and (4) Leaky ReLU (α=0.01). Record the training loss every 200 iterations for each model. After training, compute and report the final test accuracy for all four versions in a clear table. Generate and compare decision boundary contour plots for all four trained models side-by-sideAssignment details
P4: Design and implement a feedforward neural network with exactly one hidden layer from scratch using only NumPy. The network should have 2 input neurons, 48 hidden neurons, and 1 output neuron with sigmoid activation. Use the make_moons dataset from scikit-learn (n_samples=1500, noise=0.15, random_state=42) for training and a separate test set of 500 samples (random_state=43). Train the model using mini-batch gradient descent batch size 64, learning rate 0.01, for 10,000 iterations. Use Binary Cross Entropy loss.
Train four identical networks (same architecture, optimizer, hyperparameters, and random seed), changing only the hidden-layer activation function: 
(1) sigmoid, 
(2) tanh, 
(3) ReLU, and 
(4) Leaky ReLU (α=0.01). 
Record the training loss every 200 iterations for each model. After training, compute and report the final test accuracy for all four versions in a clear table.
Generate and compare decision boundary contour plots for all four trained models side-by-side


week 5:

P5: Implement two feedforward neural networks (M1: 3 hidden layers, M2: 4 hidden layers) from scratch in NumPy on attached obesity_data.csv. Compare performance across loss functions and gradient descent variants using backpropagation.
Models:
M1: 6 → 64 → 32 → 16 (ReLU) → 4 (softmax)
M2: 6 → 128 → 64 → 32 → 16 (ReLU) → 4 (softmax)
Losses (use both per model):
Mean Squared Error (MSE)
Categorical Cross-Entropy (CCE)
Optimizers (use both per model + loss):
Stochastic GD (batch size = 1)
Mini-batch GD (batch size = 40)
Show: 
Combined loss curves plot (8 runs)
Test performance table (accuracy + macro precision + macro recall + macro F1)
Confusion matrix for best model


week 6:

P6: Implement an interactive web-demonstration that trains a simple one hidden layer neural network for regression on synthetic data (or any regression dataset of your choice) and visualizes:
- The error/loss surface in 2D weight space
- The optimization trajectory of each optimizer on that surface
- Real-time weight updates during training
Network Architecture:
Input layer → Hidden layer (e.g., 2–10 neurons) → Output layer (1 neuron for regression)
Use MSE (Mean Squared Error) as the loss function
Optimizers:
SGD
Momentum
RMSprop
Adam
Adagrad


week 7:

P7: Compare Sigmoid, Tanh, and ReLU activation functions by implementing a one-hidden-layer network (2 → 12 → 1) on make_circles dataset (n_samples=1000, noise=0.1). Train all three versions with same hyperparameters and plot their loss curves together. Plot decision boundary for each as well. Discuss which activation converges faster and why.


week 8:

P8: Design and implement a Neural Network of your choice on dataset of your choice. (Dataset must have at least 5 input features/ attributes )
1. Demonstrate over-fitting.
2. Apply regularization techniques (in different combinations and separately):
       (i) L1 (ii) L2 (iii) Drop-out
3. Visualize performance in terms of loss and accuracy per epoch for training and validation set. Also, visualize the performance of the model on the test set (Confusion matrix, F1 Score, Precision, Recall, ROC AUC)
        (i) before applying regularization technique(s).
       (ii) after applying regularization technique(s).
