# (c) Laura Fink, 2024
# Based on https://github.com/NVlabs/nvdiffrec/blob/main/dataset/dataset_nerf.py

import os
import glob
import json

import torch
import numpy as np

from render import util

from .dataset import Dataset

###############################################################################
# NERF based dataset (synthetic)
# Uses simple obj (trimesh with one texture) to render synthetic dataset on demand
###############################################################################


import nvdiffrast.torch as dr
def load_obj(mesh_path):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    normals = np.asarray(mesh.vertex_normals).astype(np.float32)
    triangle_uvs = np.asarray(mesh.triangle_uvs).astype(np.float32)
    texture = np.asarray(mesh.textures[1])    
    
    verts = torch.Tensor(verts).cuda()
    faces = torch.Tensor(faces).to( dtype=torch.int32, device="cuda:0" )
    normals = torch.Tensor(normals).cuda()
    triangle_uvs = torch.Tensor(triangle_uvs).cuda()
    uv_idx = torch.arange(triangle_uvs.shape[0], dtype=torch.int32).to( device="cuda:0" ).view(-1,3)
    
    texture = torch.Tensor(texture).cuda().unsqueeze(0) / 255.0

    return verts, faces, triangle_uvs, uv_idx, texture

def render(glctx, mv, proj, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_world = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    pos_view = transform_pos(mv, pos_world)[None, ...]
    pos_clip = transform_pos(proj, pos_view)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolution)
    
    if enable_mip:
        texc, texd = dr.interpolate(uv, rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv, rast_out, uv_idx)
        # texc, _ = dr.interpolate(q_uv, rast_out, q_uv_idx)
        color = dr.texture(tex, texc, filter_mode='linear')
        
    pos_img, _ = dr.interpolate(pos_view, rast_out, pos_idx)
    depth =     -pos_img[:,:,:,2]

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color, depth

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    return torch.matmul(pos, t_mtx.t())

class DatasetSynthetic(Dataset):
    def __init__(self, cfg_path, FLAGS, glctx, examples=None, feat_encoder=None):
        
        self.glctx = glctx
        self.mesh = load_obj(FLAGS.input_dir +"/"+FLAGS.init_depth_dir)
        
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        if True: # self.FLAGS.local_rank == 0:
            print("DatasetSynthetic: load", cfg_path)

        
        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])
        self.srgb = FLAGS.srgb
        self.feat_enc = None

        # Determine resolution & aspect ratio
        self.resolution = FLAGS.cam_resolution #_load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path']), self.srgb).shape[0:2]
        self.scale = [1.0, 1.0]
        if self.resolution[0] != FLAGS.cam_resolution[0] or self.resolution[1] != FLAGS.cam_resolution[1]:
            print("resolution of args do not match img resolution --> rescale imgs from", self.resolution, "to", FLAGS.cam_resolution)
            self.scale = np.array(FLAGS.cam_resolution)/self.resolution    
            self.resolution = np.array(FLAGS.cam_resolution)    
        self.aspect = self.resolution[1] / self.resolution[0]

        if True: # if self.FLAGS.local_rank == 0:
            print("DatasetSynthetic: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        self.preloaded_data = {}
        if self.FLAGS.load_mode == "pre_load":
            for i in range(self.n_images):
                self.preloaded_data[i] = self._parse_frame(self.cfg, i, self.scale, feat_encoder)
                
        if feat_encoder != None and not self.FLAGS.load_mode == "pre_load":                
            self.feat_enc = feat_encoder # we only store this if we do not preload

    def _parse_frame(self, cfg, idx, scale, feat_enc):
        # Config projection matrix (static, so could be precomputed)
        proj = util.perspective_from_f_c(cfg['fl_x']*scale[1], cfg['fl_y']*scale[0], cfg['cx']*scale[1], cfg['cy']*scale[0],
                    self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1], self.resolution[1], self.resolution[0],device='cpu')

        # Load image data and modelview matrix
        mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        mv     = mv @ util.rotate_x(np.pi / 2)
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv
        img, depth    = render(self.glctx, mv.cuda(), proj.cuda(), *self.mesh, self.FLAGS.cam_resolution, False,1 )        

        idx = torch.tensor([idx], dtype=torch.int32)
        
        frame_data = {      # Add batch dimension
            'idx' :         idx[None, ...], 
            'mv' :          mv[None, ...],
            'mvp' :         mvp[None, ...],
            'p':            proj[None, ...],
            'campos' :      campos[None, ...],
            'img' :         img.cpu(),
            'depth' :       depth.cpu(),
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

def test_render(dataset: DatasetSynthetic):
    for i, test_view in enumerate(dataset):  
        depth = test_view["depth"]
        print(depth.shape)
        img = test_view["img"]
        print(img.shape)
        print(test_view["mvp"][0])
        color, _ = render(dataset.glctx, test_view["mv"][0].cuda(), test_view["p"][0].cuda(), *dataset.mesh, dataset.FLAGS.cam_resolution, False,1 )
        util.save_tensor_as_image(f"synth/{i}.png", color, 0)
        
        if i > 60: break
        

