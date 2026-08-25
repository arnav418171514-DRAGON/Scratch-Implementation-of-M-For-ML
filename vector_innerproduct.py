# without Libraries
house_features = [1200,4,5]
weights = [0.5,-0.8,1.6]
dot_product = sum(x*x for x in zip(house_features,weights))
print(f"Dot product = {dot_product}")

# using Numpy
import numpy as np
house_features_np = np.array(house_features)
weights_np = np.array(weights)
dot_product_np = np.dot(house_features_np, weights_np)
print(f"Numpy Dot product = {dot_product_np}")

# using PyTorch
import torch
x_tensor = torch.tensor(house_features, dtype=torch.float32)
y_tensor = torch.tensor(weights, dtype=torch.float32)
torch_dot = torch.dot(x_tensor, y_tensor)
print(f"Torch Dot product = {torch_dot}")