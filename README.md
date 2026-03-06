# Deep-Learning-Lab

I will upload my college weekly assignments here related to Deep Learning

week1:

P1: 
Without using inbuilt functions: 1. Implement a perceptron from scratch for binary classification. Using the perceptron learning rule for training, classify linearly separable data (of your choice) and plot the decision boundary for the perceptron after training. 2. Implement XOR using network of perceptrons. Visualize the results.


week2:

P2: 
Implement a multi-layered perceptron network with one output unit. 
1. Train it for different Boolean functions using the perceptron Learning Algorithm with thresholding logic (with no learning rate or error function).
2. Train it for different Boolean functions using the perceptron Learning Algorithm with sigmoid activation function (with no learning rate or error function).
   

week3:

P3:  
Implement the Gradient Descent algorithm from first principles (without using automatic differentiation libraries) to minimize a multi-variable objective function (f(x,y)=x^2+y^2+10 sin⁡(x)+10 cos⁡(y) ). You must visualize the optimization trajectory on a 2D Contour Plot(error surface) to demonstrate how Learning Rate(at least 3 different rates) and Initial weight(at least 3 different ) value affect convergence.


week4:

P4: Design and implement a feedforward neural network with exactly one hidden layer from scratch using only NumPy. The network should have 2 input neurons, 48 hidden neurons, and 1 output neuron with sigmoid activation. Use the make_moons dataset from scikit-learn (n_samples=1500, noise=0.15, random_state=42) for training and a separate test set of 500 samples (random_state=43). Train the model using mini-batch gradient descent batch size 64, learning rate 0.01, for 10,000 iterations. Use Binary Cross Entropy loss. Train four identical networks (same architecture, optimizer, hyperparameters, and random seed), changing only the hidden-layer activation function: (1) sigmoid, (2) tanh, (3) ReLU, and (4) Leaky ReLU (α=0.01). Record the training loss every 200 iterations for each model. After training, compute and report the final test accuracy for all four versions in a clear table. Generate and compare decision boundary contour plots for all four trained models side-by-sideAssignment details
P4: Design and implement a feedforward neural network with exactly one hidden layer from scratch using only NumPy. The network should have 2 input neurons, 48 hidden neurons, and 1 output neuron with sigmoid activation. Use the make_moons dataset from scikit-learn (n_samples=1500, noise=0.15, random_state=42) for training and a separate test set of 500 samples (random_state=43). Train the model using mini-batch gradient descent batch size 64, learning rate 0.01, for 10,000 iterations. Use Binary Cross Entropy loss.
Train four identical networks (same architecture, optimizer, hyperparameters, and random seed), changing only the hidden-layer activation function: 
(1) sigmoid, 
(2) tanh, 
(3) ReLU, and 
(4) Leaky ReLU (α=0.01). 
Record the training loss every 200 iterations for each model. After training, compute and report the final test accuracy for all four versions in a clear table.
Generate and compare decision boundary contour plots for all four trained models side-by-side
