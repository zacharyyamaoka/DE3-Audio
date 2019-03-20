

import numpy as np
import torch

# def radial_loss(h,y):
#
#     x = abs(h-y)
#     x = x % np.pi
#     return x

def radial_loss(h, y):
    x = torch.abs(h.sub(y))
    x = torch.remainder(x, np.pi)
    x = torch.sum(x)
    return x


def abs_radial_loss(h,y):

    h = torch.remainder(h, np.pi)
    x = torch.abs(h.sub(y))
    x = torch.abs(x - np.pi)
    x = np.pi - x

    return x




h = torch.tensor([np.pi])
y =  torch.tensor([0.0])
#
print("h: ", h.item(), "y: ", y.item())
print("curr: ", radial_loss(h,y).item())
print("abs: ", abs_radial_loss(h,y).item())


h = torch.tensor([2*np.pi])
print("h: ", h.item(), "y: ", y.item())
print("curr: ", radial_loss(h,y).item())
print("abs: ", abs_radial_loss(h,y).item())

h = torch.tensor([0.5*np.pi])
print("h: ", h.item(), "y: ", y.item())
print("curr: ", radial_loss(h,y).item())
print("abs: ", abs_radial_loss(h,y).item())


h = torch.tensor([np.pi])
y =  torch.tensor([np.pi])
print("h: ", h.item(), "y: ", y.item())
print("curr: ", radial_loss(h,y).item())
print("abs: ", abs_radial_loss(h,y).item())

h = torch.tensor([np.pi])
y =  torch.tensor([2*np.pi])
print("h: ", h.item(), "y: ", y.item())
print("curr: ", radial_loss(h,y).item())
print("abs: ", abs_radial_loss(h,y).item())

h = torch.tensor([-2*np.pi])
y =  torch.tensor([0.0])
print("h: ", h.item(), "y: ", y.item())
print("curr: ", radial_loss(h,y).item())
print("abs: ", abs_radial_loss(h,y).item())

h = torch.tensor([np.pi])
y =  torch.tensor([1.5*np.pi])
print("h: ", h.item(), "y: ", y.item())
print("curr: ", radial_loss(h,y).item())
print("abs: ", abs_radial_loss(h,y).item())


h = torch.tensor([np.pi])
y =  torch.tensor([40*np.pi])
print("h: ", h.item(), "y: ", y.item())
print("curr: ", radial_loss(h,y).item())
print("abs: ", abs_radial_loss(h,y).item())
# assert radial_loss(-pi,zero) == np.pi
# assert radial_loss(pi,-pi) == 0

# x = torch.abs(h.sub(y))
# x = torch.remainder(x, np.pi)
# x = torch.sum(x)
