# -*-coding:utf-8-*-
import os.path
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import API.utils
class Model():

    def __init__(self, init_normal_map,gt_aolp,gt_normal,gt_mask):
        # init_normal_map: (H,W,3),(-1,1)
        # gt_aolp: (H,W), (0,pi)
        super(Model, self).__init__()
        self.mask_tensor = torch.tensor(gt_mask).reshape(-1).cuda()
        self.predict_normal_tensor = torch.tensor(init_normal_map).reshape(-1,3).cuda()
        azimuth_angles = np.arctan2(init_normal_map[:,:,1],init_normal_map[:,:,0])
        zimuth_angles = np.arccos(init_normal_map[:,:,2])
        self.predict_theta_tensor = torch.tensor(zimuth_angles).reshape(-1,1).cuda()
        self.predict_phi_tensor = torch.tensor(azimuth_angles).reshape(-1,1).cuda()
        self.predict_phi_tensor.requires_grad=True
        self.gt_aolp_tensor = torch.tensor(gt_aolp).reshape(-1,1).cuda()
        self.gt_normal_tensor = torch.tensor(gt_normal).reshape(-1,3).cuda()
        self.num_pixels = torch.sum(self.mask_tensor)
        self.k = 2.0

        # optimizer params
        self.lr = 0.1
        self.lr_scheduler = 'StepLR'
        self.step_size = 20
        self.gamma = 0.1


    def get_normal_error(self):
        # 获取法线误差，这个拿来测试，后面再写
        x = torch.sin(self.predict_theta_tensor[self.mask_tensor]) * torch.cos(self.predict_phi_tensor[self.mask_tensor])
        y = torch.sin(self.predict_theta_tensor[self.mask_tensor]) * torch.sin(self.predict_phi_tensor[self.mask_tensor])
        z = torch.cos(self.predict_theta_tensor[self.mask_tensor])
        self.predict_normal_tensor[self.mask_tensor,0] = x.reshape(-1)
        self.predict_normal_tensor[self.mask_tensor,1] = y.reshape(-1)
        self.predict_normal_tensor[self.mask_tensor,2] = z.reshape(-1)
        predict = self.predict_normal_tensor[self.mask_tensor,:] # (num_pixels,3)
        ground_truth = self.gt_normal_tensor[self.mask_tensor,:] # (num_pixels,3)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        loss_cos = cos(predict, ground_truth)
        eps = 1e-10
        loss_cos = torch.clamp(loss_cos, (-1.0 + eps), (1.0 - eps))
        loss_rad = torch.acos(loss_cos)
        loss_deg = loss_rad * (180.0 / math.pi)
        loss_deg_mean = torch.mean(loss_deg)
        return loss_deg_mean

    def get_aolp_loss(self):

        # get network output azimuth angles
        azimuth_angles = torch.remainder(self.predict_phi_tensor, torch.pi * 2)[self.mask_tensor,:]  # (0,2*pi)

        # calculate aolps
        aolp_0 = self.gt_aolp_tensor[self.mask_tensor,:] + torch.pi / 2.0
        aolp_1 = self.gt_aolp_tensor[self.mask_tensor,:] - torch.pi / 2.0
        aolp_0 = torch.remainder(aolp_0, torch.pi * 2)
        aolp_1 = torch.remainder(aolp_1, torch.pi * 2)

        # calculate eta
        eta = torch.zeros(self.num_pixels, 2).cuda()  # (num_pixels, 2)
        eta[:, 0] = torch.min(torch.abs(azimuth_angles - aolp_0).reshape(-1),
                                 torch.pi * 2 - torch.abs(azimuth_angles - aolp_0).reshape(-1))
        eta[:, 1] = torch.min(torch.abs(azimuth_angles - aolp_1).reshape(-1),
                                 torch.pi * 2 - torch.abs(azimuth_angles - aolp_1).reshape(-1))


        eta, indices = torch.min(eta, dim=1)  # (num_pixels)
        print('sum aolp_gt + pi/2:',torch.sum(indices==0))
        print('sum aolp_gt - pi/2:',torch.sum(indices==1))

        theta = 2 * eta / torch.pi
        theta = 1 - theta

        loss = torch.pow((torch.exp(-self.k * theta) - math.exp(-self.k)) / (1 - math.exp(-self.k)),2)  # (num_pixels,)


        loss = loss.sum()

        return loss
    def compute_loss(self):
        loss = self.get_aolp_loss()
        loss_deg_mean = self.get_normal_error()
        print('loss_deg_mean:',loss_deg_mean)
        return loss

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(params=[self.predict_phi_tensor], lr=self.lr)
        self.schudeler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.step_size,
                                                         gamma=self.gamma)
    def optimize(self):
        self.optimizer.zero_grad()
        loss = self.compute_loss()
        print('loss:',loss)
        loss.backward()
        self.optimizer.step()

    def showNormal(self, normals):
        # normals: (H,W,3),(-1,1)
        import matplotlib.pyplot as plt
        normals = (normals + 1) * 127.5
        plt.imshow(normals.astype(np.uint8))
        plt.show()

    def showAoLP(self,aolp):
        # aolp:(H,W),(0,pi)
        aolp = (aolp / np.pi * 255.0)
        plt.imshow(aolp.astype(np.uint8),cmap='gray')
        plt.show()
    def save_maps(self,result_path):
        pass



if __name__ == "__main__":
    gt_normal_path = '/home/robotlab/smq/SurfaceNormals/results/hemi-sphere-big/000-label.png'
    gt_aolp_path = '/home/robotlab/smq/SurfaceNormals/results/hemi-sphere-big/000-aolp-label.png'
    initial_normal_path = '/home/robotlab/smq/SurfaceNormals/results/hemi-sphere-big/000-predict.png'

    gt_normal = API.utils.rgb_loader(gt_normal_path)
    gt_aolp = API.utils.rgb_loader(gt_aolp_path)[:,:]
    initial_normal = API.utils.rgb_loader(initial_normal_path)
    gt_mask = ~(np.sum(initial_normal,axis=2) ==0)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # 将法线图从0到255转换到-1到1之间
    initial_normal = (initial_normal - 127.5) / 127.5
    gt_normal = (gt_normal - 127.5) / 127.5
    gt_aolp = gt_aolp / 255.0 * np.pi # 转化到(0,pi)区间
    model = Model(initial_normal,gt_aolp,gt_normal,gt_mask)
    model.showNormal(model.predict_normal_tensor.reshape(512,512,3).detach().cpu().numpy())
    model.showAoLP(gt_aolp)
    model.setup_optimizer()


    for i in range(0, 500):
        print('#'*30)
        print('processing %d' % i)
        model.optimize()

        print('\n\n')
    azimuth_angles = model.predict_phi_tensor.reshape(512,512).detach().cpu().numpy()
    azimuth_angles = np.remainder(azimuth_angles,np.pi *2)
    model.showAoLP(azimuth_angles)
    model.showNormal(model.predict_normal_tensor.reshape(512,512,3).detach().cpu().numpy())

