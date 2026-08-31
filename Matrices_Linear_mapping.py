x =[5.0 , 32.0]
# Weight for 2 feature and 3 neurons
w = [[0.5, -1.0],
     [2.0, 1.5],
     [-0.5, 0.0]]
# bias for the neurons
b = [0.1, -0.1, 0.3]
""" 
Every neuron looks at EVERYTHING: By calculating (Weight1 * Humidity) + (Weight2 * Temperature) + Bias, a single neuron creates a mathematical "opinion" about the combination of both inputs.
It allows for Non-Linearity: A simple linear combination is just a straight line—it can't solve complex problems. However, we calculate this combination specifically so we can pass it through an Activation Function (like ReLU or Sigmoid) immediately after. The activation function takes this straight line and "bends" it.
"""
y = []
for i in range(len(w)):
    dot_product = sum(w[i][j]*x[j] for j in range(len(x)))
    y.append(dot_product + b[i])

# Using numpy
import numpy as np
w_n = np.array(w)
x_n = np.array(x)
b_n = np.array(b)
y_n = w_n @ x_n + b_n

# using torch
import torch
import torch.nn as nn

# Define a linear mapping layer from 2 inputs to 3 outputs
layer = nn.Linear(in_features=2, out_features=3)

# temporary disabling the auto gradient of pytorch 
with torch.no_grad():
    layer.weight = nn.Parameter(torch.tensor(w))
    layer.bias = nn.Parameter(torch.tensor(b))

x_tensor = torch.tensor(x)
y_tensor = layer(x_tensor)
