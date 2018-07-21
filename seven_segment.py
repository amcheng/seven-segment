# Seven segment display neural network
# By Alex Cheng July 2018
# Adapted from YouTube content creator Siraj Raval's video "Build a Neural Net in 4 Minutes"
# This script trains a neural network to solve the classic seven segment display problem
# Usually taught as a digital design problem, solved using Karnaugh maps
# This approach trains a neural network to solve the same problem.
# Input are binary numbers and output is 

import numpy as np

nu = 1

# sigmoid squishing function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid function, x represents the sigmoid function itself
def sig_deriv(x):
    return x * (1 - x)


# define training input data
# binary representation for numbers 2-15 inclusive
X = np.array([[0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 0, 1],])
# define training
Y = np.array([[1, 1, 1, 1, 1, 1, 0],  # 2 is a prime number
              [0, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 1, 1, 0, 1],
              [1, 1, 1, 1, 0, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 1, 0, 1, 1],
              [1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 1, 1]])  # 15 is not a prime number

# create weight matrices
# np.random.seed(0)
# 4 input neurons to 5 layer 1 neurons
w0 = 2 * np.random.random((4, 6)) - 1
# 5 neurons in layer 1 to 1 final output neuron
w1 = 2 * np.random.random((6, 7)) - 1

#train neural network 250 times
for i in range(10000):
    # Calculate output neuron by propagating forword input
    l0 = X
    l1 = sigmoid(l0 @ w0)
    output = sigmoid(l1 @ w1)
    # Calculate gradient for last synapse
    # Error function is (Y-l2)**2, partial derivative below:
    d_error = -2 * (Y - output)
    out_delta = d_error * sig_deriv(output)
    # Calculate gradient for first synapse
    l1_error = out_delta @ w1.T
    l1_delta = l1_error * sig_deriv(l1)
    # Adjust weight matrices with gradients
    w1 -= nu * l1.T @ out_delta
    w0 -= nu * l0.T @ l1_delta

# print out total error and final output neurons after training
print("Error: " + str(np.mean((Y - output) ** 2)))
def print_seven_segment(x=Y[0]):
    b = '#' if(x[1] > 0.5) else ' '
    c = '#' if(x[2] > 0.5) else ' '
    e = '#' if(x[4] > 0.5) else ' '
    f = '#' if(x[5] > 0.5) else ' '
    print(' # # # # ') if(x[0] > 0.5) else print()
    for i in range(4):
        print(f+'       '+b)
    print(' # # # # ') if(x[6] > 0.5) else print()
    for i in range(4):
        print(e+'       '+c)
    print(' # # # # ') if(x[3] > 0.5) else print()


for number in output:
    print_seven_segment(number)
    print('-'*20)