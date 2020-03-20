############################################################
########          ACTUAL LSTM INFERENCE        #############
############################################################
# PACKAGES
import random
import numpy as np
from numpy import array
from numpy import loadtxt
import math
from scipy.special import softmax
# INPUTS
# Input shape = [1X6]
It = array([0,0,0,0,0,0]).reshape(1,6)
ht_1 = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,16)
Ct_1 = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,16)

# LSTM LAYER
# LSTM weights = [6X64]
# Split into Wi, Wf, Wo, Wc = 4[6X16]
lstm_kernel = loadtxt('lstm_kernel.csv', delimiter=',')
Wi, Wf, Wo, Wc = np.hsplit(lstm_kernel, 4)

# LSTM recurrent weights = [16X64]
# Split into Ui, Uf, Uo, Uc = 4[16X16]
lstm_rec_kernel = loadtxt('lstm_rec_kernel.csv', delimiter=',')
Ui, Uf, Uo, Uc = np.hsplit(lstm_rec_kernel, 4)

# LSTM bias = [1X64]
# Split into bi, bf, bo, bc = 4[1X16]
lstm_bias = loadtxt('lstm_bias.csv', delimiter=',')
bi, bf, bo, bc = np.hsplit(lstm_bias, 4)

# DENSE LAYER
# Dense weights = [320X5]
rnn_kernel = loadtxt('rnn_kernel.csv', delimiter=',')

# Dense bias = [1X5]
rnn_bias = loadtxt('rnn_bias.csv', delimiter=',')


#Print all shapes
print("Input and last output")
print(It.shape)
print(ht_1.shape)
print(Ct_1.shape)
print("*************")
print("LSTM weights")
print(lstm_kernel.shape)
print(Wi.shape)
print(Wf.shape)
print(Wo.shape)
print(Wc.shape)
print("*************")
print("LSTM Rec weights")
print(lstm_rec_kernel.shape)
print(Ui.shape)
print(Uf.shape)
print(Uo.shape)
print(Uc.shape)
print("*************")
print("LSTM bias")
print(lstm_bias.shape)
print(bi.shape)
print(bf.shape)
print(bo.shape)
print(bc.shape)
print("*************")
print("Dense weights and bias")
print(rnn_kernel.shape)
print(rnn_bias.shape)
#Define activations

#Sigmoid
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

#ReLu
def relu(x):
  for i in range(len(x)):
    for j in range(len(x[0])):
      if (x[i][j]<0):
        x[i][j]=0
      else:
        x[i][j]=x[i][j]
  return x   

#Softmax
def softmax(x): 
  e_x = np.exp(x - np.max(x)) 
  return e_x / e_x.sum(axis=0)

#Matrix multiplication function

def mat_mult(X, Y):
  for i in range(len(X)):
    for j in range(len(Y[0])):
      for k in range(len(Y)):
        result[i][j] += X[i][k] * Y[k][j]
  return result
  
  # INFERENCE MATRIX MULTIPLICATION FOR LSTM LAYER
# To compute 4 matrices of shape [1X16] called i, f, o, Co

def lstm_step(It, ht_1, Ct_1):
  # Compute i = sigmoid((Wi*It) + bi + (Ui*ht_1))
  # Expected i.shape: (1,16)
  result = np.zeros((1,16))
  mult_1 = mat_mult(It,Wi)
  result = np.zeros((1,16))
  mult_2 = mat_mult(ht_1, Ui)

  add_1 = mult_1 + bi + mult_2
  i = sigmoid(add_1)

  # Compute f = sigmoid((Wf*It) + bf + (Uf*ht_1))
  # Expected f.shape: (1,16)  
  result = np.zeros((1,16))
  mult_3 = mat_mult(It,Wf)
  result = np.zeros((1,16))
  mult_4 = mat_mult(ht_1, Uf)

  add_2 = mult_3 + bf + mult_4
  f = sigmoid(add_2)

  # Compute o = sigmoid((Wo*It) + bo + (Uo*ht_1))
  # Expected o.shape: (1,16)
  result = np.zeros((1,16))
  mult_5 = mat_mult(It,Wo)
  result = np.zeros((1,16))
  mult_6 = mat_mult(ht_1, Uo)

  add_3 = mult_5 + bo + mult_6
  o = sigmoid(add_3)

  # Compute Co = relu((Wc*It) + bc + (Uc*ht_1))
  # Expected Co.shape: (1,16)
  result = np.zeros((1,16))
  mult_7 = mat_mult(It,Wc)
  result = np.zeros((1,16))
  mult_8 = mat_mult(ht_1, Uc)

  add_4 = mult_7 + bc + mult_8
  Co = relu(add_4)

  # Compute Ct = ((i*Co) + (f*Ct_1)) // * - hadamard product
  # Compute ht = (o*relu(Ct)) // * - hadamard product
  # Expected ht.shape and Ct.shape = [1X16]
  mult_9 = np.multiply(i,Co)
  mult_10 = np.multiply(f, Ct_1)
  Ct = mult_9 + mult_10

  relu_Ct = relu(Ct)
  ht = np.multiply(o,relu_Ct)

  return ht, Ct, ht_1, Ct_1
