y_true = [3.0, 5.5, 2.0]
y_pred = [2.8, 5.0, 2.5]
squared_errors = []
for true, pred in zip(y_true, y_pred):
    error = true - pred
    squared_errors.append(error**2)
mse = sum(squared_errors)/len(squared_errors)

# using numpy
import numpy as np
y_true_n = np.array(y_true)
y_pred_n = np.array(y_pred)
mse = np.mean((y_true_n - y_pred_n)**2)
# Calculating the L2 Norm (Euclidean length)
error_vector = y_true_n - y_pred_n
l2_norm = np.linalg.norm(error_vector)

# using Torch
import torch
import torch.nn as nn

y_t_tensor = torch.tensor(y_true)
y_p_tensor = torch.tensor(y_pred)

# Instantiate the standard Mean Squared Error loss function
loss_function = nn.MSELoss()

# Calculate the loss
loss = loss_function(y_p_tensor, y_t_tensor)
print(f"PyTorch MSE Loss: {loss.item()}")
