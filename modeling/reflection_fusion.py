import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectionFusion(nn.Module):
    def __init__(self,neighbor_size = 9):
        super(ReflectionFusion, self).__init__()
        self.neighbor_size = neighbor_size
        self.padding = int((self.neighbor_size-1)/2)

    def forward(self, x,p):
        filter = torch.ones(3,3,self.neighbor_size,self.neighbor_size)/self.neighbor_size**2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        filter = filter.to(device)
        mean_normal = F.conv2d(x,filter,stride=1,padding=self.padding)
        res = p*x + (1-p)*mean_normal
        return res

if __name__ == '__main__':
    x = torch.ones([8,3,500,500])
    p = torch.ones([8,1,500,500])


    model = ReflectionFusion()
    res = model(x,p)