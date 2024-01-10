import torch
import numpy as np

tensor = torch.rand(3,4)
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


#Standard numpy slicing and indexing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0 #replace second column with 0
print(tensor)

#Joining tensors along a given dimension
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z2)
print(z3)

#Single element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))