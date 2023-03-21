import torch
import torch.nn as nn

from core.utils.network_util import initseq
import torch.nn.functional as F
import math


class voxel_model(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, input_dir_ch=8,xyz_min = [-1.5, -1.5, -1.5] , xyz_max = [1.5, 1.5, 1.5],
                 input_ch=3, skips=None,num_voxels=None,k0_dim=None,
                 **_):
        super(voxel_model, self).__init__()

        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch
        
        self.xyz_min = torch.tensor(xyz_min)
        self.xyz_max = torch.tensor(xyz_max)
        self.init = False
        num_voxels = num_voxels**3
        self._set_grid_resolution(num_voxels)
        self.k0_dim = k0_dim
        self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        self.k0_pre_scene = torch.nn.Parameter(torch.zeros([1, self.k0_dim*1, *self.world_size]))




    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        device = xyz.device

        if self.xyz_min.device != device:
            self.xyz_min=self.xyz_min.to(device)
            self.xyz_max=self.xyz_max.to(device)


        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def mult_dist_interp(self, ray_pts_delta, grid):

        x_pad = math.ceil((grid.shape[2]-1)/4.0)*4-grid.shape[2]+1
        y_pad = math.ceil((grid.shape[3]-1)/4.0)*4-grid.shape[3]+1
        z_pad = math.ceil((grid.shape[4]-1)/4.0)*4-grid.shape[4]+1
        grid_pad = F.pad(grid.float(),(0,z_pad,0,y_pad,0,x_pad))
        # three 
        vox_l = self.grid_sampler(ray_pts_delta, grid_pad)
        vox_m = self.grid_sampler(ray_pts_delta, grid_pad[:,:,::2,::2,::2])
        vox_s = self.grid_sampler(ray_pts_delta, grid_pad[:,:,::4,::4,::4])
        vox_feature = torch.cat((vox_l,vox_m,vox_s),-1)

        if len(vox_feature.shape)==1:
            vox_feature_flatten = vox_feature.unsqueeze(0)
        else:
            vox_feature_flatten = vox_feature
        
        return vox_feature_flatten

    def forward(self,  xyz,subject_id=None):

        k0 = self.mult_dist_interp(xyz,self.k0)
        k0_pre_scene = self.mult_dist_interp(xyz ,self.k0_pre_scene)
        return k0 , k0_pre_scene
        