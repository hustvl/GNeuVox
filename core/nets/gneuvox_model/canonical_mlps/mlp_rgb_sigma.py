import torch
import torch.nn as nn

from core.utils.network_util import initseq
import torch.nn.functional as F


class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, input_dir_ch=8,input_time_ch=17,
                 input_ch=3, skips=None,input_voxel_ch=None,
                 **_):
        super(CanonicalMLP, self).__init__()



        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_voxel_ch = input_voxel_ch

        self.input_ch = input_ch
        self.fast_color_thres = 1e-4

        alpha_init = torch.tensor(1e-2)
        self.act_shift = torch.log(1/(1-alpha_init) - 1)
        self.k0_dim = self.input_voxel_ch

        rgbnet_width = mlp_width
        rgbnet_depth = 3
        # dim0 = input_dir_ch + self.k0_dim 

        input_dim = self.k0_dim + input_ch +input_time_ch
        featurenet_width = mlp_width
        featurenet_depth = 2
        self.featurenet = nn.Sequential(
            nn.Linear(input_dim + self.k0_dim, featurenet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(featurenet_width, featurenet_width), nn.ReLU(inplace=True))
                for _ in range(featurenet_depth-1)
            ],
            )

        self.densitynet = nn.Linear(featurenet_width, 1)
        dim0 = input_dir_ch + featurenet_width
        
        self.rgbnet = nn.Sequential(
            nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                for _ in range(rgbnet_depth-2)
            ],
            nn.Linear(rgbnet_width, 3),
        )



    # def forward(self, k0,xyz_embedded, dir_embedded,k0_pre_scene,**_):
    #     # print(k0.shape)
    #     rgb_feat = torch.cat([k0, k0_pre_scene ,xyz_embedded], -1)
    #     feature = self.featurenet(rgb_feat)
    #     rgb_logit = self.rgbnet(torch.cat([feature, dir_embedded], -1) )
    #     density =self.densitynet(feature)
    #     outputs = torch.cat([rgb_logit, (density+self.act_shift)], -1)
    #     # outputs = self.output_linear(h)

    #     return outputs    


    def forward(self, k0,xyz_embedded, dir_embedded,k0_pre_scene,time_embedded,**_):
        # print(k0.shape)
        rgb_feat = torch.cat([k0, k0_pre_scene ,xyz_embedded ,time_embedded ], -1)
        feature = self.featurenet(rgb_feat)
        density =self.densitynet(feature)
        
        mask = (density>=self.fast_color_thres)
        # mask = mask.float()
        mask = mask.squeeze(1)    # mask
        density[mask==0] = 0
        dir_embedded_mask = dir_embedded[mask]
        feature_mask = feature[mask]

        rgb_logit = self.rgbnet(torch.cat([feature_mask, dir_embedded_mask], -1) )
        # density =self.densitynet(feature)
        rgb_logit_all = torch.zeros([density.shape[0],3],device=density.device)
        rgb_logit_all[mask] = rgb_logit

        outputs = torch.cat([rgb_logit_all, (density+self.act_shift)], -1)
        return outputs    