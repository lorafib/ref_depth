# Main Edits by Laura Fink:
#   * Different loading schemes added
#   * Feature encoder + caching added
#   * Return values changed


# Based on https://github.com/NVlabs/nvdiffrec/blob/main/dataset/dataset_nerf.py
# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import json

import torch
import numpy as np

from render import util

from .dataset import Dataset

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path, srgb=True):
    files = glob.glob(path + '.*')
    if os.path.isfile(path): 
        files += [path]
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img, dtype=torch.float32)  / 255.0
        if srgb:
            img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

class DatasetNERF(Dataset):
    def __init__(self, cfg_path, FLAGS, examples=None, feat_encoder=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        if True:
            print("DatasetNERF: load", cfg_path)

        
        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])
        self.srgb = FLAGS.srgb
        self.feat_enc = None

        # Determine resolution & aspect ratio
        self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path']), self.srgb).shape[0:2]
        self.scale = [1.0, 1.0]
        if self.resolution[0] != FLAGS.cam_resolution[0] or self.resolution[1] != FLAGS.cam_resolution[1]:
            print("resolution of args do not match img resolution --> rescale imgs from", self.resolution, "to", FLAGS.cam_resolution)
            self.scale = np.array(FLAGS.cam_resolution)/self.resolution    
            self.resolution = np.array(FLAGS.cam_resolution)    
        self.aspect = self.resolution[1] / self.resolution[0]

        if True: 
            print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        self.preloaded_data = {}
        if self.FLAGS.load_mode == "pre_load":
            for i in range(self.n_images):
                self.preloaded_data[i] = self._parse_frame(self.cfg, i, self.scale, feat_encoder)
                
        if feat_encoder != None and not self.FLAGS.load_mode == "pre_load":                
            self.feat_enc = feat_encoder # we only store this if we do not preload
            
    def get_idx_of_frame(self, name):
        for idx, f in enumerate(self.cfg["frames"]):
            if name in f["file_path"]:
                return idx
        return None

    def _parse_frame(self, cfg, idx, scale, feat_enc):
        # Config projection matrix (static, so could be precomputed)
        proj = util.perspective_from_f_c(cfg['fl_x']*scale[1], cfg['fl_y']*scale[0], cfg['cx']*scale[1], cfg['cy']*scale[0],
                    self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1], self.resolution[1], self.resolution[0],device='cpu')

        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']), self.srgb)
        img    = util.scale_img_hwc(img, (self.resolution[0], self.resolution[1]))
        mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        mv = util.cam_convention_transform(mv, self.FLAGS.cam_use_flip_mat, use_rot_mat = True) 
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        idx = torch.tensor([idx], dtype=torch.int32)
        
        frame_data = {      # Add batch dimension
            'idx' :         idx[None, ...], 
            'mv' :          mv[None, ...],
            'mvp' :         mvp[None, ...],
            'p':            proj[None, ...],
            'campos' :      campos[None, ...],
            'img' :         img[None, ...],
            'resolution' :  self.FLAGS.cam_resolution
        }

        # encode if not during preload phase
        if feat_enc != None:
            frame_data["feat"] = torch.cat(feat_enc.forward_nhwc(frame_data["img"].cuda()), dim=-1).detach()
            
        if len(self.preloaded_data) == self.n_images-1:
            print("\ndataset nerf: Lazy load done!\n")
        
        return frame_data 

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.cam_resolution
        
        img      = []
        fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        idx = itr % self.n_images
        
        if self.FLAGS.load_mode == "on_demand": 
            return self._parse_frame(self.cfg, idx, self.scale)
        elif self.FLAGS.load_mode == "lazy_load": 
            if not idx in self.preloaded_data:
                self.preloaded_data[idx] = self._parse_frame(self.cfg, idx, self.scale, self.feat_enc)
            return self.preloaded_data[idx]
        elif self.FLAGS.load_mode == "pre_load":
            return self.preloaded_data[idx]
        else:
            print("datast nerf error: invalid load  mode")
