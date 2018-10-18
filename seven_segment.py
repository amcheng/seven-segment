# Seven segment display neural network
# By Alex Cheng July 2018
# Adapted from YouTube content creator Siraj Raval's video "Build a Neural Net in 4 Minutes"
# This script trains a neural network to solve the classic seven segment display problem.
# The 7-segment problem takes 4 digital inputs, representing a binary number 0-9,
# and maps them to 7 digital outputs, each representing a segment on the display
# Usually taught as a digital design problem and solved using Karnaugh maps.
# I studied this problem in college, and I thought I could solve it using a neural network too.
# try it!

import numpy as np

# gradient descent rates, learning rate
nu = [1.0,2.0,5.0]

def sigmoid(x):
  """sigmoid function maps [-inf,inf] to [0,1]"""
    return 1 / (1 + np.exp(-x))


def sig_deriv(x):
  """sigmoid function derivative where x is sigmoid function itself"""
    return x * (1 - x)


def print_seven_seg(x):
  """Prints an individual number given a list of 7 node values"""
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



def print_all_numbers(out_layer):
  """Prints to console an entire set of 10 numbers 0-9 given a list of lists"""
  for num in out_layer:
    print(' # # # #      ', end='') if(num[0] > 0.5) else print('              ', end='')
  print()
  for i in range(4):
    for num in out_layer:
      b = '#' if(num[1] > 0.5) else ' '
      f = '#' if(num[5] > 0.5) else ' '
      print(f+'       '+b+'     ', end='')  
    print()
  for num in out_layer:
    print(' # # # #      ', end='') if(num[6] > 0.5) else print('              ', end='')
  print()
  for i in range(4):
    for num in out_layer:
      c = '#' if(num[2] > 0.5) else ' '
      e = '#' if(num[4] > 0.5) else ' '
      print(e+'       '+c+'     ', end='')
    print()
  for num in out_layer:
    print(' # # # #      ', end='') if(num[3] > 0.5) else print('              ', end='')
  print()



# define training data
# input, binary representation for numbers 0-9
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
# outputs, drawings of numbers represented by 
# activating a segment or keeping it dark, each list draws a number
Y = np.array([[1, 1, 1, 1, 1, 1, 0],
              [0, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 1, 1, 0, 1],
              [1, 1, 1, 1, 0, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 1, 0, 1, 1],
              [1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 1, 1]])

def main():
    # seed random generator for consistency between runs
    np.random.seed(1)
    # creating weight matrices, 2 inner layers and output layer
    # 4 input, 6 output neurons
    w0 = 2 * np.random.random((4, 6)) - 1
    # 6 input, 6 output neurons
    w1 = 2 * np.random.random((6, 6)) - 1
    # 6 input, 7 output neurons (output layer)
    w2 = 2 * np.random.random((6, 7)) - 1

    #train neural network
    for i in range(100):
        # Calculate output neuron by propagating forward input
        l0 = X
        l1 = sigmoid(l0 @ w0)
        l2 = sigmoid(l1 @ w1)
        output_layer = sigmoid(l2 @ w2)
        input("------- Press enter to train neural network -------")
        print_all_numbers(output_layer)
        # for number in output_layer:
        #   print_seven_seg(number)
        #   print('-'*10)

        # Calculate gradient for last synapse
        # Error function is (Y-output)**2, partial derivative below:
        dE_dout = -2*(Y - output_layer)
        delta_output = dE_dout * sig_deriv(output_layer)
        # Calculate gradient for first synapse
        dE_dl2  = delta_output @ w2.T
        delta_l2 = dE_dl2 * sig_deriv(l2)
        # Adjust the weight matrices with gradients
        dE_dl1 = delta_l2 @ w1.T
        delta_l1 = dE_dl1 * sig_deriv(l1)
        # calculate gradients for each weight matrix and descend.  Uses learning rate for coefficient
        w2 -= nu[2] * l2.T @ delta_output
        w1 -= nu[1] * l1.T @ delta_l2
        w0 -= nu[0] * l0.T @ delta_l1

    # print out final output layer after training
    for number in output_layer:
        print_seven_seg(number)
        print('-'*10)
    print("Average error: " + str(np.mean((Y - output_layer) ** 2)))

if __name__ == '__main__':
    main()
     