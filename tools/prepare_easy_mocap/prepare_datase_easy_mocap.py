import os
import sys

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image
import cv2
from absl import app
from absl import flags
FLAGS = flags.FLAGS
import shutil

flags.DEFINE_string('cfg',
                    '387.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'

import pickle
import numpy as np


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def get_camera(camera_path):
    camera = read_pickle(camera_path)
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1
    R = np.eye(3)
    T = np.zeros([3])
    D = camera['camera_k']
    camera = {'K': K, 'R': R, 'T': T, 'D': D}
    return camera

def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name,subject):
    # print('get_mask',subject_dir, img_name)
    msk_path = os.path.join(subject_dir ,'masks',
                            'frame_'+subject+'_'+img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk[msk == 1] = 255

    return msk


def main(argv):
    del argv  # Unused.
    cfg = parse_config()
    subject = cfg['dataset']['subject']
    dataset_dir = cfg['dataset']['zju_mocap_path']
    subject_dir = os.path.join(cfg['dataset']['zju_mocap_path'], f"{subject}")
    sex = cfg['dataset']['sex']
    print('subject',subject)
    max_frames = cfg['max_frames']
    print('max_frames',max_frames)

    max_frames = cfg['max_frames']


    # load image paths
    img_paths = os.path.join(subject_dir,'images', '0')
    img_paths_all = os.listdir(img_paths)
    img_paths_all.sort(key=lambda x: int(x[:-4]))


    output_path = os.path.join(cfg['output']['dir'],subject)
    os.makedirs(output_path, exist_ok=True)
    out_img_dir  = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')
    


    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
    smpl_params_dir = os.path.join(subject_dir, 'output-smpl-3d/smplfull/0')


    cameras = {}
    mesh_infos = {}
    all_betas = []
    for idx, ipath in enumerate(tqdm(img_paths_all)):

        img_path = os.path.join(img_paths, ipath)
        print(ipath)
        # load image
        img = np.array(load_image(img_path))
        img_name = ipath.split('.')[0]
        out_name = 'frame_'+subject+'_{:06d}'.format(int(img_name))
        
        params_path = os.path.join(smpl_params_dir, "{:0>6}.json".format(img_name))


        with open(params_path,'r',encoding = 'utf-8') as fp:
            smpl_params = json.load(fp)
   
        fp.close()


        betas = np.array(smpl_params['annots'][0]['shapes'][0]) #(10,)
        poses = np.array(smpl_params['annots'][0]['poses'][0]) #(72,)
        Rh = np.array(smpl_params['annots'][0]['Rh'][0] ) #(3,)
        Th = np.array(smpl_params['annots'][0]['Th'][0])#(3,)

        
        all_betas.append(betas)

        K = np.array(smpl_params['K'])     #(3, 3)
        D = np.zeros(5)
        E = np.eye(4)  #(4, 4)
        E[:3, :3] = smpl_params['R']  
        E[:3, 3]= smpl_params['T'][0]
        # write camera info
        cameras[out_name] = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': D
        }

        # write mesh info
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        _, joints = smpl_model(poses, betas)
        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints, 
            'tpose_joints': tpose_joints
        }

        # load and write mask
        mask = get_mask(subject_dir, ipath,subject)
        save_image(to_3ch_image(mask), 
                   os.path.join(out_mask_dir, out_name+'.png'))

        # write image
        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)

    # write camera infos
    with open(os.path.join(output_path, 'cameras_'+subject+'.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos_'+subject+'.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints_'+subject+'.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints_'+subject: template_joints,
            }, f)

if __name__ == '__main__':
    app.run(main)
