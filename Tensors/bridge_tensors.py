import torch
import numpy as np


#Tensor to numpy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#Change in tensor reflects in numpy array
t.add_(1) #in-place derivative
print(f"t: {t}")
print(f"n: {n}")

#Numpy array to tensor
n = np.ones(4)
t = torch.from_numpy(n)

#Change in numpy array reflects in tensor
np.add(n, 3, out=n)
print(f"t: {t}")
print(f"n: {n}")