import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Gradient_Net(nn.Module):
  def __init__(self,device):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x)
    grad_y = F.conv2d(x, self.weight_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    return gradient
# class Gradient_Net(nn.Module):
#   def __init__(self,device):
#     super(Gradient_Net, self).__init__()
#     kernel_x = [[0., 0., 0.],[0., 0., 1.],  [0., 0., 0.]]
#     kernel_x = torch.DoubleTensor(kernel_x).unsqueeze(0).unsqueeze(0)
#     kernel_x = kernel_x.expand(3,3,3,3)
#     kernel_x = kernel_x.to(device)
#
#     kernel_y = [[0., 0., 0.],[0., 0., 0.],  [0., 1., 0.]]
#     kernel_y = torch.DoubleTensor(kernel_y).unsqueeze(0).unsqueeze(0)
#     kernel_y = kernel_y.expand(3,3,3,3)
#     kernel_y = kernel_y.to(device)
#
#     self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
#     self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
#     self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#
#   def forward(self, x):
#     # x_1_y = nn.functional.conv2d(x, self.weight_x,padding=3//2)
#     # x_y_1 = nn.functional.conv2d(x, self.weight_y,padding=3//2)
#     # temp = x_1_y.detach().cpu().numpy()
#     # temp = (temp + 1)/2
#     # plt.imshow(temp[0,:,:,:].transpose(1,2,0))
#     # plt.show()
#     # gradient_x = 1 - self.cos(x_1_y,x)
#     # gradient_y = 1 - self.cos(x_y_1,x)
#     h_x = x.size()[2]
#     w_x = x.size()[3]
#     r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
#     l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
#     t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
#     b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
#     grad_x = 1 - self.cos(r,l)
#     grad_y = 1 - self.cos(t,b)
#     # temp = grad_x.detach().cpu().numpy()
#     # plt.imshow(temp[0,:,:])
#     # plt.show()
#
#     gradient = torch.abs(grad_x) + torch.abs(grad_y)
#     gradient[torch.where(gradient>1)] = 0.0
#     # temp = gradient.detach().cpu().numpy()
#     # plt.imshow(temp[0,:,:])
#     # plt.show()
#     return gradient