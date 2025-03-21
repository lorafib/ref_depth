# Copyright 2024, Laura Fink

from pathlib import Path

import torch
import numpy as np
import nvdiffrast.torch as dr
import slangtorch

from dataset.cv_dataset_utils.conversion import scene
from dataset.cv_dataset_utils.depth_maps.build_mesh_from_depthmap import create_o3d_mesh, simplify_mesh, np_from_o3d_mesh

from . import util
from .depth_image import DepthImage
from models.mlp import DepthTransferFunc



this_file_path = str(Path(__file__).parent.resolve())
print(this_file_path)
_calc_tri_indices_module = slangtorch.loadModule(this_file_path+"/slang_utils/calc_tri_indices.slang", verbose=True)

near_far_eps = 0.0001

# Convert depth image to 3D points
def positions_from_depth_map(depth_image, fx, fy, cx, cy, img_width, img_height, down_scale, impl=torch):
    Z = depth_image[::down_scale, ::down_scale]
    # tri_height, tri_width = Z.shape 
    tri_height, tri_width = img_height//down_scale, img_width//down_scale
    if Z.shape[0] != tri_height or Z.shape[1] != tri_width:
        print(Z.shape, "is not equal to", tri_height,tri_width) 
    xx, yy = impl.meshgrid(impl.arange(tri_width), impl.arange(tri_height), indexing='xy')
    xx = xx*down_scale
    yy = yy*down_scale
    
    if impl == torch:
        xx = xx.to(device=Z.device)
        yy = yy.to(device=Z.device)
    
    X = (xx - cx + 0.5) * Z / fx  # + 0.5 seems to be necessary to align with gl convention  
    Y = (yy - cy + 0.5) * Z / fy  # + 0.5 seems to be necessary to align with gl convention
    Z = -Z # for gl convention
    Y = -Y # for gl convention
    
    # Flatten the X, Y, Z coordinates into a (N, 3) array
    points = impl.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    return points

def construct_triangles_slang(points, tri_width, tri_height, cam_near_far):
    num_tris = 2 * (tri_width - 1) * (tri_height - 1)
    triangles = torch.empty(num_tris, 3, dtype=torch.int, device='cuda')
    t_points = torch.from_numpy(points).to(dtype=torch.float32, device="cuda").contiguous()
    _calc_tri_indices_module.calc_tri_indices_kernel(pos=t_points, width=tri_width, height=tri_height, near=cam_near_far[0]+near_far_eps, far=cam_near_far[1]-near_far_eps, output=triangles).launchRaw(blockSize=(32, 32, 1), gridSize=((tri_width-1+31)//32,(tri_height-1+31)//32, 1))
    return triangles.cpu().numpy()

def are_vaild_points(pos, near, far):
    valid_depth = (np.all(np.abs(pos[:,2]) < far))
    valid_depth = valid_depth and (np.all(np.abs(pos[:,2]) > near))
    valid_values = (not np.any(np.isnan(pos))) and (not np.any(np.isinf(pos)))
    return valid_depth and valid_values

def construct_triangles_py(points, tri_width, tri_height, cam_near_far):
    # Loop through each pixel except for the last row and column
    triangles = []
    for y in range(tri_height - 1):
        for x in range(tri_width - 1):
            idx = y * tri_width + x
            
            # Triangle 1
            idx1 = [idx, idx + 1, idx + tri_width]     
            if are_vaild_points(points[idx1], cam_near_far[0]+near_far_eps, cam_near_far[1]-near_far_eps):
                triangles.append(idx1)
            else:
                triangles.append([-1,-1,-1])
                
            # Triangle 2
            idx2 = [idx + 1, idx + 1 + tri_width, idx + tri_width]
            if are_vaild_points(points[idx2], cam_near_far[0]+near_far_eps, cam_near_far[1]-near_far_eps):
                triangles.append(idx2)
            else:
                triangles.append([-1,-1,-1])
    return np.array(triangles)

def depth_to_mesh(depth_image,fx, fy, cx, cy, cam_near_far, mesh_downscale=1, slang=True):
    """
    Convert an RGB-D image to a mesh.

    Parameters:
    - depth_image: A 2D NumPy array with depth values.
    - fx, fy: Focal lengths of the camera in pixels.
    - cx, cy: Optical center of the camera in pixels.

    Returns:
    - points: A 2D NumPy array (Nx3) with XYZ coordinates of points.
    - triangles: A 2D NumPy array (Nx3) with vertex indices forming triangles.
    - uvs: A 2D NumPy array (Nx2) with texture coordinates for the points.
    """
    img_height, img_width = depth_image.shape
    tri_height, tri_width = depth_image[::mesh_downscale, ::mesh_downscale].shape
    
    points = positions_from_depth_map(depth_image, fx, fy, cx, cy, img_width, img_height, mesh_downscale, impl=np)
    
    if slang:
        triangles = construct_triangles_slang(points, tri_width, tri_height, cam_near_far)
    else:
        triangles = construct_triangles_py(points, tri_width, tri_height, cam_near_far)
            
    xx, yy = np.meshgrid(np.arange(tri_width), np.arange(tri_height))
    xx = xx/float(tri_width) +0.5/float(img_width)
    yy = yy/float(tri_height) +0.5/float(img_height)
    uvs = np.vstack((xx.flatten(), yy.flatten())).T
                        
    return points, triangles, uvs

# Transform vertex positions to clip space
def transform_pos(mtx, posw: torch.tensor):
    """
    transform positions by 4x4 mat

    Parameters:
    - mtx: transform matrix in row major [b, 4, 4]
    - posw: [N, 4]

    Returns: transformed points expanded for b
    """
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    t_mtx = t_mtx.permute(0,2,1) 
    return torch.matmul(posw[None, ...].expand(t_mtx.shape[0], -1, -1), t_mtx)


class RGBD_Mesh():
    def __init__(self, args, glctx, depth_image, scene:scene.BaseScene, view, down_scale, mode="eval", color_image=None, simplification_ratio=None):
        
        self.glctx = glctx

        # get original camera parameters
        self.fx, self.fy, self.cx, self.cy = view.camera.fx, view.camera.fy, view.camera.cx, view.camera.cy
        self.width, self.height = view.camera.width, view.camera.height
        self.color_image = color_image
        if type(color_image) == type(None):
            self.color_image    = util.load_image(args.input_dir/scene.color_path/view.image_name)
        
        # scale intrinsics to match user chosen resolutuion (nvdiffrast demands image resolutions divisible by 32)
        self.img_scale = [1.0, 1.0]
        if self.height != args.cam_resolution[0] or self.width != args.cam_resolution[1]:
            print("resolution of args do not match img resolution --> rescale imgs from", self.height, self.width, "to", args.cam_resolution)
            self.fx, self.fy, self.cx, self.cy, self.width, self.height, self.img_scale = util.scale_intrinsics(self.fx, self.fy, self.cx, self.cy, self.width, self.height, args.cam_resolution[1], args.cam_resolution[0])

        # gl style projection matrix
        self.near, self.far = args.cam_near_far[0], args.cam_near_far[1]
        self.proj = util.perspective_from_f_c(  self.fx, self.fy, self.cx, self.cy, self.near, self.far, 
                                                self.width, self.height, "cpu").numpy()
        # how many vertices do we skip initially
        self.mesh_downscale = down_scale
        
        # depth 
        self.depth_image = depth_image

        # construct mesh from depth map
        self.pos, self.pos_idx, self.uv = depth_to_mesh(self.depth_image.get_absolute.cpu().numpy(), self.fx, self.fy, self.cx, self.cy, args.cam_near_far, mesh_downscale=self.mesh_downscale)
       
        # matrices + to tensors
        self.proj   = torch.tensor(self.proj, dtype=torch.float32).cuda()
        self.mv     = torch.tensor(view.W2C(), dtype=torch.float32).cuda()
        self.mv     = util.cam_convention_transform(self.mv, args.cam_use_flip_mat, use_rot_mat = True)
        self.cam2world = torch.linalg.inv(self.mv)
        self.mvp    = torch.matmul(self.proj, self.mv)[None, ...]

        # postions to tensors
        self.color_image    = torch.tensor(self.color_image, dtype=torch.float32).cuda()[None, ...]
        self.color_image    = util.scale_img_nhwc(self.color_image, (self.height, self.width))
        self.pos_idx        = torch.tensor(self.pos_idx, dtype=torch.int32).cuda()
        self.uv             = torch.tensor(self.uv, dtype=torch.float32).cuda().contiguous()
        self.pos            = torch.tensor(self.pos, dtype=torch.float32).cuda()
        # (x,y,z) -> (x,y,z,1)
        self.pos = torch.cat([self.pos, torch.ones([self.pos.shape[0], 1]).cuda()], axis=1)
        self.pos = torch.matmul(self.pos,self.cam2world.t())
        
        # add additional idx tensors
        self.uv_idx = self.pos_idx.clone().detach()
        self.pos_idx_hash = dr.antialias_construct_topology_hash(self.pos_idx) # assumes tri indices are constant!

        # neural "texture" of image 
        # only used with feat_loss
        self.c_feat_image = None # TODO? currently set in main!
        self.f_feat_image = None

        # setup train mode
        self.is_optimized = None
        self.mode = mode
        
        if self.mode == "train_color":
            self.color_image = torch.randn_like(self.color_image)
            
              
            
    def pos_from_dm(self):
        return positions_from_depth_map(self.depth_image.get_absolute, self.fx, self.fy, self.cx, self.cy, self.width, self.height, self.mesh_downscale)

    def render(self, glctx, mtx, resolution, out_attr:list = ["color"], enable_mip=False, max_mip_level=0, mv=None):
        out = {}
        if self.mode == "train_depth" or self.mode == "train_scale":
            self.pos = self.pos_from_dm()
            self.pos = torch.cat([self.pos, torch.ones([self.pos.shape[0], 1]).cuda()], axis=1)
            cam2world = torch.linalg.inv(self.mv)
            self.pos = torch.matmul(self.pos,cam2world.t())

        pos_clip = transform_pos(mtx, self.pos)
        
        # rasterize      
        rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, self.pos_idx, resolution=resolution)
        out["mask"] = torch.clamp(rast_out[..., -1:], 0, 1)
        
        # interpolate texture coords  
        texc, texd = None, None
        if enable_mip:
            texc, texd = dr.interpolate(self.uv, rast_out, self.uv_idx, rast_db=rast_out_db, diff_attrs='all')
            tex_filter_mode='linear-mipmap-linear'
            tex_max_mip_level=max_mip_level
        else:
            texc, _ = dr.interpolate(self.uv, rast_out, self.uv_idx)
            tex_filter_mode='linear'
            tex_max_mip_level=None
        
        # sample textures
        if "color" in out_attr:
            color = dr.texture(self.color_image, texc, texd, filter_mode=tex_filter_mode, max_mip_level=tex_max_mip_level)            
            color = color * out["mask"] # Mask out background.
            color = dr.antialias(color, rast_out, pos_clip, self.pos_idx, topology_hash=self.pos_idx_hash)
            out["color"]  = color
        if "feat" in out_attr:
            c_feat_img = dr.texture(self.c_feat_image, texc, texd, filter_mode=tex_filter_mode, max_mip_level=tex_max_mip_level)
            f_feat_img = dr.texture(self.f_feat_image, texc, texd, filter_mode=tex_filter_mode, max_mip_level=tex_max_mip_level)
            feat_img = torch.cat([c_feat_img,f_feat_img],dim=-1) * out["mask"] # Mask out background.
            out["feat"] = feat_img
        if "wpos" in out_attr:
            pos_img, _ = dr.interpolate(self.pos, rast_out, self.pos_idx)
            out["wpos"] = pos_img
        if "vpos" in out_attr or "depth" in out_attr:
            pos_img, _ = dr.interpolate(torch.matmul(self.pos.unsqueeze(0), mv.permute(0,2,1)), rast_out, self.pos_idx)
            out["vpos"] = pos_img
            depth =     out["vpos"].clone()
            depth =     -depth[0,:,:,2]
            out["depth"] = depth   
        if "rast_out" in out_attr:
            out["rast_out"] = rast_out
            
            
        return out
    
    def render_self(self, out_attr=["depth"]):
        out = {}
        attrs = out_attr
        out =  self.render(self.glctx, self.mvp, [self.height, self.width], out_attr=attrs, enable_mip=False, max_mip_level=0, mv=self.mv.unsqueeze(0))
                
        return out

def simplify_mesh_in_viewspace(simplification_ratio, pos, pos_idx, uv, proj, color_image):
    """
    decimates a mesh via open3d
    inputs are numpy arrays, outputs are tensors!

    Parameters:

    Returns: 
    """
    # mesh simplification via open3d
    o3d_mesh = create_o3d_mesh(pos, pos_idx)
    o3d_mesh = simplify_mesh(o3d_mesh, simplification_ratio)
    pos, pos_idx, _ = np_from_o3d_mesh(o3d_mesh)
    # recalc of uvs happens later
    
    # positions
    pos = torch.tensor(pos, dtype=torch.float32, ).cuda()
    # (x,y,z) -> (x,y,z,1)
    pos = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    
    # recalc uvs
    uv_pos = torch.matmul(pos,proj.t())
    # print( pos_clip.shape)
    uv_pos = uv_pos/uv_pos[:,3,None] # dehomo
    uv = torch.clamp(0.5*uv_pos[:,:2]+0.5, 0,1) # map to 0,1
    # uv = uv_pos[:,:2] 
    # print(pos.shape, pos_idx.shape, uv.shape)
    return pos, pos_idx, uv

def simplify_mesh_in_invz(simplification_ratio, pos, pos_idx, uv, proj, color_image):
    """
    decimates a mesh via open3d
    inputs are numpy arrays, outputs are tensors!

    Parameters:

    Returns: 
    """
    # inverse z
    pos[:,2] = 1./pos[:,2]
    pos[:,:2] = pos[:,:2]*pos[:,2,None]
    # mesh simplification via open3d
    o3d_mesh = create_o3d_mesh(pos, pos_idx)
    o3d_mesh = simplify_mesh(o3d_mesh, simplification_ratio)
    pos, pos_idx, _ = np_from_o3d_mesh(o3d_mesh)
    # recalc of uvs happens later
    # inverse z again
    pos[:,2] = 1./pos[:,2]
    pos[:,:2] = pos[:,:2]*pos[:,2,None]

    
    # positions
    pos = torch.tensor(pos, dtype=torch.float32, ).cuda()
    # (x,y,z) -> (x,y,z,1)
    pos = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    
    # recalc uvs
    uv_pos = torch.matmul(pos,proj.t())
    # print( pos_clip.shape)
    uv_pos = uv_pos/uv_pos[:,3,None] # dehomo
    uv = torch.clamp(0.5*uv_pos[:,:2]+0.5, 0,1) # map to 0,1
    # uv = uv_pos[:,:2] 
    # print(pos.shape, pos_idx.shape, uv.shape)
    return pos, pos_idx, uv


def simplify_mesh_in_clipspace(simplification_ratio, clip_pos, pos_idx, uv, proj, color_image):
    """
    decimates a mesh via open3d
    inputs are numpy arrays, outputs are tensors!

    Parameters: all inputs are numpy arrays

    Returns: all outputs are numpy arrays
    """
    
    if np.any(np.isnan(clip_pos)): 
        print("simplify_mesh_in_clipspace error: nan encountered")
        exit()
    if np.any(np.isinf(clip_pos)): 
        print("simplify_mesh_in_clipspace error: inf encountered")
        exit()
    
    # mesh simplification via open3d
    if simplification_ratio < 1.0:
        print("create o3d mesh")
        o3d_mesh = create_o3d_mesh(clip_pos, pos_idx)
        print("done")
        # simplify mesh, strips uvs
        o3d_mesh = simplify_mesh(o3d_mesh, simplification_ratio)
        clip_pos, pos_idx, _ = np_from_o3d_mesh(o3d_mesh)
    else: print("keep full res mesh")
    
    # uvs are stripped during simplification, are recovered from clip pos
    # uvs map to 0,1
    uv = np.clip(0.5*clip_pos[:,:2]+0.5, 0,1) 
    
    # (x,y,z) -> (x,y,z,1)
    clip_pos = np.concatenate([clip_pos, np.ones_like(clip_pos[:,:1])], axis=1)
    # transform to view space
    # to view
    view_pos = np.matmul(clip_pos,np.linalg.inv(proj).T)
    # dehomo,(x,y,z,1) -> (x,y,z)
    view_pos = view_pos/view_pos[:,3,None] 
    
    return clip_pos, view_pos, pos_idx, uv

def calc_clippos_w_uv_offset(xy11, uv_offset, mesh_downscale, width, height, clamp_to=0.25):
    """_summary_

    Args:
        xy11 (_type_): xy of the clip pos aka image coordinates in [-1,1]
        uv_offset (_type_): optmized xy offsets  
        mesh_downscale (_type_): how many vertices do are skipped initially (from args)
        width (_type_): img width
        height (_type_): img height
        clamp_to (float, optional): In pixels. Defaults to 0.25. 
            for 2-stage optimization use clamp_to=0.25: 
                we allow half the way during init phase and the second half during diect depth optim
            for a single optmization pass use clamp_to=0.50: 
                we allow half of the minimum distance between two vertices. 

    Returns:
        updated xy of the clip pos in [-1,1] nd uvs in [0,1]
    """    
 

    # option 1: limit uv offset by mapping between -1,1 and then scale to 0.5*(mesh_downscale)/wh
    # limited_uv_offset = torch.tanh(uv_offset) # limit to -1,1
    # wh = torch.tensor([width, height], dtype=torch.float32).cuda()
    # limited_uv_offset = 0.5*(mesh_downscale)/wh * limited_uv_offset # allow offset up to mesh_downscale/2 pixels
    
    # option 2: simply clamp to 0.5*(mesh_downscale)/wh it offset gets to large (converges faster)
    wh = torch.tensor([width, height], dtype=torch.float32).cuda()

    limited_uv_offset = torch.clamp(uv_offset, -clamp_to*(mesh_downscale)/wh, clamp_to*(mesh_downscale)/wh)
    # print(" WARNING, tris might flip, set scale to 0.5 again!\n", "uvset in px", torch.std_mean(limited_uv_offset*wh), torch.min(limited_uv_offset*wh), torch.max(limited_uv_offset*wh))
    
    xy11 = xy11 + limited_uv_offset
    # pos = pos + torch.cat([limited_uv_offset, z_offset_nonlin, torch.zeros_like(limited_uv_offset[:,:1])], dim=-1)                        # with z_offset_nonlin
    xy11 = torch.clamp(xy11, -1, 1) 
    uv = 0.5*xy11+0.5
    return xy11, uv


class Adaptive_RGBD_Mesh():
    def __init__(self, args, glctx, depth_image, scene:scene.BaseScene, view, down_scale, mode="eval", color_image=None, simplification_ratio=None):
        
        self.glctx = glctx

        # get original camera parameters
        self.fx, self.fy, self.cx, self.cy = view.camera.fx, view.camera.fy, view.camera.cx, view.camera.cy
        self.width, self.height = view.camera.width, view.camera.height
        self.color_image = color_image
        if type(color_image) == type(None):
            self.color_image    = util.load_image(args.input_dir/scene.color_path/view.image_name)
        
        # scale intrinsics to match user chosen resolutuion (nvdiffrast demands image resolutions divisible by 32)
        self.img_scale = [1.0, 1.0]
        if self.height != args.cam_resolution[0] or self.width != args.cam_resolution[1]:
            print("resolution of args do not match img resolution --> rescale imgs from", self.height, self.width, "to", args.cam_resolution)
            self.fx, self.fy, self.cx, self.cy, self.width, self.height, self.img_scale = util.scale_intrinsics(self.fx, self.fy, self.cx, self.cy, self.width, self.height, args.cam_resolution[1], args.cam_resolution[0])

        # gl style projection matrix
        self.near, self.far = args.cam_near_far[0], args.cam_near_far[1]
        self.proj = util.perspective_from_f_c(  self.fx, self.fy, self.cx, self.cy, self.near, self.far, 
                                                self.width, self.height, "cpu").numpy()
        # how many vertices do we skip initially
        self.mesh_downscale = down_scale
        
        # depth 
        self.depth_image = depth_image
           
        # get edge map [N, C, H, W] --> [N, H, W, C]
        self.edge_map = calc_edge_map(self.depth_image.get_absolute.unsqueeze(0).unsqueeze(0)).permute(0,2,3,1)
        
        # construct mesh from depth map
        view_pos, pos_idx, uv = depth_to_mesh(self.depth_image.get_absolute.cpu().numpy(), self.fx, self.fy, self.cx, self.cy, args.cam_near_far, mesh_downscale=self.mesh_downscale)
        
        # transfrom to clip space
        # (x,y,z) -> (x,y,z,1)
        pos = np.concatenate([view_pos, np.ones([view_pos.shape[0], 1])], axis=1)
        # to clip
        clip_pos = np.matmul(pos,self.proj.T)
        # dehomo,(x,y,z,1) -> (x,y,z) 
        clip_pos[:,3, None][np.abs(clip_pos[:,3, None]) < 1e-6] = np.inf
        clip_pos = clip_pos[:,:3]/clip_pos[:,3, None] 
        clip_pos = np.clip(clip_pos, -1., 1.)
        
        # simplify mesh
        self.clip_pos, _, self.pos_idx, self.uv = simplify_mesh_in_clipspace(simplification_ratio, clip_pos, pos_idx, uv, self.proj, self.color_image)
        # simplify mesh with inverse z; looks very similar to clip space decimation but does not preserve border vertices well!
        # self.pos, self.pos_idx, self.uv = simplify_mesh_in_invz(simplification_ratio, self.pos, self.pos_idx, self.uv, self.proj, self.color_image)

        # postions to tensors
        self.color_image    = torch.tensor(self.color_image, dtype=torch.float32).cuda()[None, ...]
        self.color_image    = util.scale_img_nhwc(self.color_image, (self.height, self.width))
        self.clip_pos       = torch.tensor(self.clip_pos, dtype=torch.float32).cuda()
        self.pos_idx        = torch.tensor(self.pos_idx, dtype=torch.int32).cuda()
        self.uv             = torch.tensor(self.uv, dtype=torch.float32).cuda().contiguous()
        # add additional idx tensors
        self.uv_idx = self.pos_idx.clone().detach()
        self.pos_idx_hash = dr.antialias_construct_topology_hash(self.pos_idx) # assumes tri indices are constant!

        # for depth optimization
        self.uv_offset          = torch.zeros_like(self.clip_pos[:,:2])
        self.z_offset_nonlin    = torch.zeros_like(self.clip_pos[:,:1])
        self.z_factor           = torch.ones_like(self.clip_pos[:,:1])
        
        # matrices + to tensors
        self.proj   = torch.tensor(self.proj, dtype=torch.float32).cuda()
        self.mv     = torch.tensor(view.W2C(), dtype=torch.float32).cuda()
        self.mv     = util.cam_convention_transform(self.mv, args.cam_use_flip_mat, use_rot_mat = True)
        self.cam2world = torch.linalg.inv(self.mv)
        self.mvp    = torch.matmul(self.proj, self.mv)[None, ...]
        
        # neural "texture" of image 
        # only used with feat_loss
        self.c_feat_image = None # TODO? currently set in main!
        self.f_feat_image = None
                
        self.set_train_mode(args, mode)

    def set_train_mode(self, args, mode):
         # setup train mode
        self.mode = mode

        # for optim of offset and scale
        self.offset_scale = None
        self.offset_scale = torch.tensor([0.0,1.0]).cuda()  # default
        if "transfer_func" in self.mode:
            print("init tranfer func mlp")
            self.depth_transfer_function = DepthTransferFunc(args, self.depth_image.offset_scale).cuda()
            
        # self.offset_scale = torch.tensor([0.2091, 1.1910]).cuda()  # optimized for train
        if self.mode == "train_color":
            self.color_image = torch.randn_like(self.color_image)
            
        elif self.mode == "train_scale": 
            self.offset_scale = torch.tensor([0.0,1.0]).cuda()  # default
    
    def cuda(self):
        for v in self.__dict__.values():
            if type(v) == torch.Tensor:
                v.cuda()
    
    @ torch.no_grad()   
    def bake_init_transfer_func_uv_alignment(self):
        # get current values
        vpos = self.get_vpos() # sets self.uv
        if torch.any(vpos[:,3] <= self.near):
            print("WARNING: z got smaller or equal to 0! might fail during baking", flush=True)
        # to clip
        clippos = torch.matmul(vpos,self.proj.t())
        # dehomo
        clippos = clippos/clippos[:,3,None] 
        
        # update
        self.clip_pos = torch.clamp(clippos, -1., 1.)
   
        # reset stuff
        self.depth_transfer_function = None
        self.uv_offset = torch.zeros_like(self.uv_offset)
        self.mode = None
    
    @ torch.no_grad()    
    def simplifiy_mesh(self, simplification_ratio):
        # stuff to simplify        
        simpl_args = [self.clip_pos[:,:3], self.pos_idx, self.uv, self.proj, self.color_image]
        # cuda --> cpu --> np
        simpl_args = [sa.cpu().numpy() if type(sa) == torch.Tensor else sa for sa in simpl_args]            
        
        # simplify
        self.clip_pos, _, self.pos_idx, self.uv = simplify_mesh_in_clipspace(simplification_ratio, *simpl_args) #, pos_to_vspace= not("uv" in mode))
        
        # np --> cpu --> cuda
        self.clip_pos        = torch.tensor(self.clip_pos, dtype=torch.float32).cuda()
        self.pos_idx    = torch.tensor(self.pos_idx, dtype=torch.int32).cuda()
        self.uv         = torch.tensor(self.uv, dtype=torch.float32).cuda().contiguous()
        
        # re-init optmizable parameters    
        if self.uv_offset.max() > 0.: 
            print("CAUTION: uv offset result are reset but were not empty!")    
        self.uv_offset          = torch.zeros_like(self.clip_pos[:,:2])
        self.z_offset_nonlin    = torch.zeros_like(self.clip_pos[:,:1])
        self.z_factor           = torch.ones_like(self.clip_pos[:,:1])
        # and other index things
        self.uv_idx = self.pos_idx.clone().detach()
        self.pos_idx_hash = dr.antialias_construct_topology_hash(self.pos_idx) # assumes tri indices are constant!
        
    
    def update_depth_image_from_pos(self): 
        '''
        Must be called once per frame to update the rasterized depth map and calculate regularizers and so on.
        '''
        # print(self.depth_image.offset_scale)
        depth = self.render_self(["depth"])["depth"]
        self.depth_image.from_absolute_depth_offset_scale(depth, self.depth_image.offset_scale)

    def render_self(self, out_attr=["depth"]):
        out = {}
        attrs = out_attr
        out =  self.render(self.glctx, self.mvp, [self.height, self.width], out_attr=attrs, enable_mip=False, max_mip_level=0, mv=self.mv.unsqueeze(0))
        
        if "depth" in out: 
            out["depth"] = out["depth"][0]
        return out
    
    
    def get_pos_w_uv_offset(self): 
        xy11, uv  = calc_clippos_w_uv_offset(self.clip_pos[:,:2], self.uv_offset, self.mesh_downscale, self.width, self.height)
        pos = torch.cat([xy11, self.clip_pos[:,2:]], dim=-1) # witth z_factor
        return pos, uv


    def get_vpos(self):
        
        # calc current clip pos and uvs using offset
        # if uv is not optmized, zeros will be added
        pos, uv = self.get_pos_w_uv_offset()
        self.uv = uv
        
        # pos = self.pos
        # to view
        pos = torch.matmul(pos,torch.linalg.inv(self.proj).t())
        # dehomo
        pos = pos[:,:3]/pos[:,3,None] 
        
        # apply offset and scale from rgbd mesh (!= offset and scale from depth image!)
        # will be 0 and 1 if unsused
        pos = pos+util.safe_normalize(pos)*self.offset_scale[0]
        pos = pos*torch.clamp_min(self.offset_scale[1], 1e-12) # Z_factor should not go below or equal to 0

        if "transfer_func" in self.mode:
            # apply scale and offset from transfer func
            z =  pos[:,2,None]  
            positive_tranferred_depth = self.depth_transfer_function(self.uv, z)
            pos = (pos/z)*(-positive_tranferred_depth)
            
        if "depth" in self.mode:
            pos = pos/torch.clamp_min(self.z_factor, 1e-12) # Z_factor should not go below or equal to 0 
            
            
        # setting z between near and far
        # split
        xy = pos[:,:2]
        z =  pos[:,2,None]  
        
        # if we only clip z but not xy, we will "leave" the ray and might end up with uvs oustide of -1,1 --> instable!
        # far/z will be smaller one if z exceeds far --> will scale back to far plane
        far_factor = torch.clamp_max(-(self.far-0.001*self.far)/z, 1.0)
        xy  = xy*far_factor
        z   = z*far_factor
        # near/z will be bigger than one if z is smaller than near --> will scale back to near plane
        near_factor = torch.clamp_min(-(self.near+1e-6)/z, 1.0)
        xy  = xy*near_factor
        z   = z*near_factor
        
        # put back together
        pos = torch.cat([xy, z], dim=1)    
        # homogeneous coord        
        pos = torch.cat([pos, torch.ones_like(z)], dim=1)
        
        return pos
     
        
    def get_wpos(self):
        pos = self.get_vpos()
        return torch.matmul(pos,self.cam2world.t())


    def render(self, glctx, mtx, resolution, out_attr:list = ["color"], enable_mip=False, max_mip_level=0, mv=None):
        out = {}

        # get current world space positions
        world_pos = self.get_wpos()
        
        if (torch.any(world_pos.isinf()) or torch.any(world_pos.isnan())):
            print("invalid values for pos inf / nan: ", torch.any(world_pos.isinf()), torch.any(world_pos.isnan()))
            # exit()
                
        # transform to view
        pos_clip = transform_pos(mtx, world_pos)
        # no dehomo here (seems to make things instable)
      
        # rasterize      
        rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, self.pos_idx, resolution=resolution)
        out["mask"] = torch.clamp(rast_out[..., -1:], 0, 1)
        
        # interpolate texture coords  
        texc, texd = None, None
        if enable_mip:
            texc, texd = dr.interpolate(self.uv, rast_out, self.uv_idx, rast_db=rast_out_db, diff_attrs='all')
            tex_filter_mode='linear-mipmap-linear'
            tex_max_mip_level=max_mip_level
        else:
            texc, _ = dr.interpolate(self.uv, rast_out, self.uv_idx)
            tex_filter_mode='linear'
            tex_max_mip_level=None
            
        # sample textures
        if "color" in out_attr:
            color = dr.texture(self.color_image, texc, texd, filter_mode=tex_filter_mode, max_mip_level=tex_max_mip_level)            
            color = color * out["mask"] # Mask out background.
            color = dr.antialias(color, rast_out, pos_clip, self.pos_idx, topology_hash=self.pos_idx_hash)
            out["color"]  = color
        if "edge_map" in out_attr:
            edge_map = dr.texture(self.edge_map, texc, texd, filter_mode=tex_filter_mode, max_mip_level=tex_max_mip_level)
            out["edge_map"] = torch.clamp(edge_map, 0.,1.)
        if "feat" in out_attr:
            c_feat_img = dr.texture(self.c_feat_image, texc, texd, filter_mode=tex_filter_mode, max_mip_level=tex_max_mip_level)
            f_feat_img = dr.texture(self.f_feat_image, texc, texd, filter_mode=tex_filter_mode, max_mip_level=tex_max_mip_level)
            feat_img = torch.cat([c_feat_img,f_feat_img],dim=-1) * out["mask"] # Mask out background.
            out["feat"] = feat_img
        # gbuffer-like data
        if "grad" in out_attr:
            pos_img, _ = dr.interpolate(self.z_factor.grad, rast_out, self.pos_idx)
            out["grad"] = pos_img
        if "vignette" in out_attr:
            vignette = torch.clamp((-(1.1*self.clip_pos[:,:2])**4+1).min(dim=-1, keepdim=True).values, 0.,1.)
            pos_img, _ = dr.interpolate(vignette, rast_out, self.pos_idx)
            out["vignette"] = pos_img
        if "wpos" in out_attr:
            pos_img, _ = dr.interpolate(world_pos, rast_out, self.pos_idx)
            out["wpos"] = pos_img
        if "vpos" in out_attr or "depth" in out_attr:
            pos_img, _ = dr.interpolate(torch.matmul(world_pos.unsqueeze(0), mv.permute(0,2,1)), rast_out, self.pos_idx)
            out["vpos"] = pos_img
            depth =     out["vpos"].clone()
            depth =     -depth[:,:,:,2]
            out["depth"] = depth  
        if "rast_out" in out_attr:
            out["rast_out"] = rast_out
        if "uv" in out_attr:
            texc, _ = dr.interpolate(self.uv, rast_out, self.uv_idx)
            out["uv"] = texc
        # for visualization
        if "wireframe" in out_attr:
            barys = rast_out[...,:2]
            barys = torch.cat([barys, torch.ones_like(barys[...,0,None])- barys[...,0,None]- barys[...,1,None]], dim=-1)
            barys, _ = torch.min(barys, dim=-1, keepdim=True)
            barys[barys > 0.075] = 1.0
            barys[(barys > 0.03) & (barys < 0.075) ] = 0.5
            out["wireframe"] = barys 
            
        return out
    
# calc edge map stuff
blur = util.GaussBlur(1,7)
sobel = util.Sobel(window_size=3).sobel
def calc_edge_map(t, weight_scale=5):
    # Weights need to be [out_channels, in_channels, height, width]
    
    # Convolution
    t = torch.clamp_min(t,0.01)
    filtered = sobel(t)
    
    # see loss.py
    filtered = (weight_scale*torch.abs(filtered/(t)))    
    
    # Calculate the magnitude of gradients
    gradient_magnitude = 1-torch.sqrt(torch.sum(filtered**2, dim=1, keepdim=True)+1e-6)
    gradient_magnitude = blur.blur(gradient_magnitude)

    return gradient_magnitude
