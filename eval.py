import skimage
import os

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from configs import cfg, args

from third_parties.lpips import LPIPS
import imageio
EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height']
lpips = LPIPS(net='vgg')#.cuda()

def load_network():
    model = create_network()
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image

def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


def get_loss(rgb, target):
    lpips_loss = lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                       scale_for_lpips(target.permute(0, 3, 1, 2)))
    return torch.mean(lpips_loss).cpu().detach().numpy()

def evaluate():
    cfg.perturb = 0.

    psnr_l = []
    ssim_l = []
    lpips_l = []

    model = load_network()
    test_loader = create_dataloader('eval_cam')
    save_imgaes_path = cfg.logdir + '/eval_images_'+cfg.load_net+'_white'
    os.makedirs(save_imgaes_path, exist_ok=True)
    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        pred_img, alpha_img, gt_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])
        imageio.imwrite(save_imgaes_path+'/'+str(idx).zfill(4)+'_pre.png', pred_img)
        imageio.imwrite(save_imgaes_path+'/'+str(idx).zfill(4)+'_gt.png', gt_img)

        pred_img_norm = pred_img / 255.
        gt_img_norm = gt_img / 255.

        # get evaluation
        psnr_l.append(psnr_metric(pred_img_norm, gt_img_norm))
        ssim_l.append(skimage.metrics.structural_similarity(pred_img_norm, gt_img_norm, multichannel=True))
        lpips_loss = get_loss(rgb=torch.from_numpy(pred_img_norm).float().unsqueeze(0), target=torch.from_numpy(gt_img_norm).float().unsqueeze(0))
        lpips_l.append(lpips_loss)
        

    return psnr_l, ssim_l, lpips_l

psnr_l, ssim_l, lpips_l = evaluate()
print (np.array(psnr_l).mean())
print (np.array(ssim_l).mean())
print (np.array(lpips_l).mean())