from ast import Num
# from time import time
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.nets.gneuvox_model.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_refine, \
    load_voxel

from configs import cfg


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)
        # for pose deformation

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )
        # pose decoder MLP
        self.pose_refine = \
            load_pose_refine(cfg.pose_refine.module)(
                embedding_size=cfg.pose_refine.embedding_size,
                mlp_width=cfg.pose_refine.mlp_width,
                mlp_depth=cfg.pose_refine.mlp_depth)
        
        # canonical positional encoding
        # for xyz
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn
        # for dir
        cnl_dir_embed_fn, cnl_dir_embed_size = \
            get_embedder(cfg.canonical_mlp.dir_multires, 
                         cfg.canonical_mlp.i_embed)
        self.dir_embed_fn = cnl_dir_embed_fn
        # for voxel_feature 
        voxel_embed_fn, voxel_embed_size = \
            get_embedder(cfg.voxel.voxel_multires,input_dims=cfg.voxel.k0_dim*3)
        self.voxel_embed_fn = voxel_embed_fn

        # canonical mlp 
        time_embed_fn, time_embed_size = \
        get_embedder(cfg.canonical_mlp.time_multires,input_dims=1)
        self.time_embed_fn = time_embed_fn
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size, 
                input_dir_ch = cnl_dir_embed_size,
                input_voxel_ch = voxel_embed_size,
                input_time_ch = time_embed_size,
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips)

        # voxel 
        skips = [4]
        self.voxel= \
            load_voxel(cfg.voxel.module)(
                k0_dim=cfg.voxel.k0_dim,
                num_voxels = cfg.voxel.num_voxels,
                skips=skips)

    


    def _query_mlp(
            self,
            pos_xyz,
            ob_xyz,
            rays_d_br,
            pos_embed_fn, 
            dir_embed_fn,
            subject_id,time_id):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        obpos_flat = torch.reshape(ob_xyz, [-1, ob_xyz.shape[-1]])
        rays_d_flat = torch.reshape(rays_d_br, [-1, pos_xyz.shape[-1]])

        # print('same',pos_flat.shape , rays_d_flat.shape )
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        obpos_flat=obpos_flat,
                        rays_d_flat =rays_d_flat ,
                        pos_embed_fn=pos_embed_fn,
                        dir_embed_fn = dir_embed_fn,
                        time_id =time_id,subject_id=subject_id,
                        chunk=chunk)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            obpos_flat,
            rays_d_flat,
            pos_embed_fn,
            dir_embed_fn,
            subject_id,time_id,
            chunk):
        raws = []

        
        for i in range(0, pos_flat.shape[0], chunk):
            # aaa = time.time()
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]

            xyz = pos_flat[start:end]
            dir = rays_d_flat[start:end]
            obpos= obpos_flat[start:end]



            xyz_embedded = pos_embed_fn(xyz)
            obxyz_embedded = pos_embed_fn(obpos)

            dir_embedded = dir_embed_fn(dir)

            k0 , k0_pre_scene =self.voxel(xyz=xyz,subject_id=subject_id.item())


            k0_embedded  = self.voxel_embed_fn(k0)
            k0_pre_scene_embedded  = self.voxel_embed_fn(k0_pre_scene)

            time_id_ = torch.ones((k0_embedded.shape[0],1),device = k0_embedded.device)*time_id
            time_embedded  = self.time_embed_fn(time_id_)


            raws += [self.cnl_mlp(k0_embedded,obxyz_embedded, dir_embedded,k0_pre_scene=k0_pre_scene_embedded,time_embedded=time_embedded)]


        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0])

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):

        def _raw2alpha(raw, dists, act_fn=F.relu):
            act_fn = F.softplus
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

        return rgb_map, acc_map, weights, depth_map


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]                                     
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_max_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            dir_embed_fn,
            smpl_Th,
            smpl_R,
            bgcolor=None,
            time_id =None,subject_id=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        rays_d_br = rays_d.unsqueeze(1) + torch.zeros_like(pts)

        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']
        # bbb = time.time()
        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                rays_d_br = rays_d_br,
                                ob_xyz = pts,
                                pos_embed_fn=pos_embed_fn,
                                dir_embed_fn = dir_embed_fn,
                                time_id =time_id,subject_id=subject_id,
                                )
        raw = query_result['raws']
        rgb_map, acc_map, _, depth_map = \
            self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)


        return {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map}


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):
        # aaaa = time.time()
        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose

        pose_out = self.pose_refine(dst_posevec)
        refined_Rs = pose_out['Rs']
        refined_Ts = pose_out.get('Ts', None)
        
        dst_Rs_no_root = dst_Rs[:, 1:, ...]
        dst_Rs_no_root = self._multiply_corrected_Rs(
                                    dst_Rs_no_root, 
                                    refined_Rs)
        dst_Rs = torch.cat(
            [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

        if refined_Ts is not None:
            dst_Ts = dst_Ts + refined_Ts


        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            'dir_embed_fn': self.dir_embed_fn,
        })
        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })
        # ccc= time.time()
        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)
        # ddd = time.time()

        # print("all " ,ddd-aaaa ,'redenr',ddd-ccc ,'mweight_vol_decoder',ccc-bbb)
         # all  0.07725858688354492 redenr 0.06960606575012207 mweight_vol_decoder 0.005251884460449219
        return all_ret
