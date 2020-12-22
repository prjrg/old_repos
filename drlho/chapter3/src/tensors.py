import torch
import numpy as np

a = torch.FloatTensor(3, 2)
a.zero_()

print(torch.FloatTensor([[1,2,3],[3,2,1]]))

n = np.zeros(shape=(3,2))
print(n)

b = torch.tensor(n)

a = torch.tensor([1, 2, 3])
print(a)

print(a.sum())

print(a.sum().item())

v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])

v_sum = v1 + v2
v_res = (v_sum * 2).sum()

print(v_res)

