import torch
import numpy as np
import imageio
import sys
import cv2
import matplotlib


load_depth_scale=0.001
def load_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)

def load_dm(fn, load_depth_scale=0.001) -> np.ndarray:
    return (load_depth_scale*load_image_raw(fn).astype(np.float64)).astype(np.float32)

        
def save_dm(fn, x : np.array, save_depth_scale=1000.0):
    # np_depth = np.clip(np_depth, 0, 100) # for now clamp to 100m
    x = save_depth_scale * x
    # Save depth images (convert to uint16)
    cv2.imwrite(str(fn), x.astype(np.uint16))
    
cmap = matplotlib.colormaps.get_cmap('Spectral')
def dm_color_map(x: np.array, min, max):
    x = np.clip((x - min)/(max-min), 0,1)
    mask = np.zeros_like(x)
    mask[x.nonzero()] = 1.0
    x = (cmap(x)[:, :, :3] )
    x =  mask[...,None]*x
    return x

def save_dm_color(fn, x: np.array, min, max):
    x = dm_color_map(x,min, max)
    print(x.shape)
    cv2.imwrite(str(fn), (x* 255).astype(np.uint8)[:, :, ::-1])
        

def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    if not( (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] <= size[0] and x.shape[2] <= size[1])):
        print("WARNING: Trying to magnify image in one dimension and minify in the other")
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC
   
def calc_metrics(args, d_opt, d_ref, mask_ref):
    # resize to reference
    d_opt = scale_img_hwc(d_opt.unsqueeze(-1),d_ref.shape[0:2],min="nearest-exact", mag="nearest-exact").squeeze(-1)
    
    torch.Tensor()
    
    if torch.any(torch.isnan(d_opt)) or torch.any(torch.isinf(d_opt)): print("calc_metrics: nan or inf in d_opt detected")
    
    metrics = {}
    
    mask = mask_ref.clone()
    mask[d_ref < 1e-12] = 0.
    numel_total = mask.numel() 
    numel_ref = mask.nonzero().shape[0]
    mask_opt = torch.ones_like(d_opt)
    mask_opt[d_opt < 1e-12] = 0.
    
    numel_opt = mask_opt.nonzero().shape[0]
    metrics["#opt/#ref"] = numel_opt / numel_ref 
    metrics["#opt/#total"] = numel_opt / numel_total
    metrics["#ref/#total"] = numel_ref / numel_total
    
    mask = mask_opt*mask
        
    # calc metrics by comparing agains referencs depth
    def d_rmse_of_clamped(d_opt, d_ref, mask, clamp_to):
        mask_clamped = mask.to(dtype=bool)
        initmaskshape = mask_clamped.nonzero().shape[0]
        mask_clamped[d_ref > clamp_to] = False # only use depth values which are smaller than the clamp value
        d_opt_clamped = torch.clamp_max(d_opt, clamp_to)[mask_clamped]
        d_ref_clamped = torch.clamp_max(d_ref, clamp_to)[mask_clamped]

        if initmaskshape == 0 : 
            d_opt_clamped = d_opt
            d_ref_clamped = d_ref
            print("d_rmse_of_clamped: DM all zero!!")
        
        depth_rmse = torch.sqrt(torch.nn.functional.mse_loss(d_opt_clamped, d_ref_clamped))
        
        depth_mae = torch.mean(torch.abs(d_opt_clamped - d_ref_clamped))
        # see relative absolute error from https://arxiv.org/pdf/1612.02401.pdf 
        L1_rel = torch.mean(torch.abs(d_opt_clamped - d_ref_clamped) / d_ref_clamped)
        # see inverse absolute error from https://arxiv.org/pdf/1612.02401.pdf 
        L1_inv = torch.mean(torch.abs(1./d_opt_clamped - 1./d_ref_clamped))
        
        # see depth anything https://arxiv.org/pdf/2401.10891 
        delta_1_ = (torch.max((d_opt_clamped/d_ref_clamped).unsqueeze(-1), (d_ref_clamped/d_opt_clamped).unsqueeze(-1)) < 1.25).to(dtype=torch.float32)
        delta_1_ = delta_1_.mean()
        
        if initmaskshape == 0 : 
            L1_inv = torch.tensor([torch.nan])
        
        return depth_rmse, depth_mae, L1_rel, L1_inv, delta_1_
    for dmax in [4.0, 7.0, 10.0, 15.0]:
        depth_rmse, depth_mae, L1_rel, L1_inv, delta_1 = d_rmse_of_clamped(d_opt, d_ref, mask, dmax)
        metrics[f"depth_rmse (consider up to {dmax}m)"]     = depth_rmse.item()
        metrics[f"depth_mae (consider up to {dmax}m)"]      = depth_mae.item()
        metrics[f"depth_L1_rel (consider up to {dmax}m)"]   = L1_rel.item()
        metrics[f"depth_L1_inv (consider up to {dmax}m)"]   = L1_inv.item()
        metrics[f"depth_delta_1(consider up to {dmax}m)"]   = delta_1.item()
        
    def d_acc(d_opt, d_ref, mask, threshold):
        abs_diff = np.abs(d_opt -d_ref)
        correct = abs_diff < threshold
        mask = mask
    
        # Accuracy adopted for depth values from
        # eth3d paper: Accuracy is defined as the fraction of reconstruction points which are within a distance threshold of the ground truth points. 
        accuracy = np.sum(correct & mask) / np.sum(mask)
        
        # Depth Map Error Ratio
        # Assuming a simple ratio of mean absolute error to a baseline error threshold
        mean_abs_error = np.mean(abs_diff[mask])
        depth_error_ratio = mean_abs_error / threshold
        
        return accuracy, depth_error_ratio
    
    d_opt = d_opt.cpu().numpy() 
    d_ref = d_ref.cpu().numpy() 
    mask = mask_ref.to(dtype=bool).cpu().numpy() # reset mask to ref mask
    for tau in [0.01, 0.05, 0.1]:
        accuracy, depth_error_ratio = d_acc(d_opt, d_ref, mask, tau)
        metrics[f"depth_accuracy (tau={tau})"]    = accuracy.item()
        metrics[f"depth_error_ratio (tau={tau})"] = depth_error_ratio.item()
        
    return metrics
    
    
import numpy as np
from scipy.spatial import cKDTree as KDTree

def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(d_opt, pcd_ref):
    """_summary_

    Args:
        d_opt (_type_): the depth map that should be evaluated
        pcd_ref (_type_): the reference pcd

        NOTE: both are assumed to be in 3D and being in the same space / coordinate frame
    Returns:
        _type_: _description_
    """    
    if type(pcd_ref) != KDTree:
        gt_points_kd_tree = KDTree(pcd_ref)
    else: gt_points_kd_tree = pcd_ref
    distances, idx = gt_points_kd_tree.query(d_opt.reshape(-1,3))
    acc = np.mean(distances)

    acc_median = np.median(distances)


    return acc, acc_median


    
    
# if __name__ == "__main__":
    
#     dm_opt = load_dm(sys.argv[1])
#     dm_ref = load_dm(sys.argv[2])
    
#     dm_opt = torch.tensor(dm_opt, dtype=torch.float32).cuda()
#     dm_ref = torch.tensor(dm_ref, dtype=torch.float32).cuda()

    
#     mask = torch.zeros_like(dm_ref)  # Add channel dimension to mask for broadcasting
#     mask[dm_ref.nonzero(as_tuple=True)] = 1.0
#     # mask = mask[...,0]
    
#     print(calc_metrics(None, dm_opt, dm_ref, mask))



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
    # Z = -Z # for gl convention
    # Y = -Y # for gl convention
    
    # Flatten the X, Y, Z coordinates into a (N, 3) array
    points = impl.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    return points

if __name__ == "__main__":
    
    
    import open3d as o3d
    
    import sys
    try:
        sys.path.append("..")
        from conversion.scene import *
        import conversion.colmap  as colmap
    except:
        from ..conversion.scene import *
        from ..conversion import colmap


            
    print("read pcd", flush=True)
    pcd_ref = o3d.io.read_point_cloud(sys.argv[2])
    pcd_ref = pcd_ref.voxel_down_sample(0.0025)
    print("done", flush=True)
    pcd_ref = np.array(pcd_ref.points)

    print("Compute kdtree", flush=True)
    pcd_tree = KDTree(pcd_ref)
    print("done", flush=True)
    
    min_gt = np.array([pcd_ref[:,0].min(), pcd_ref[:,1].min(), pcd_ref[:,2].min()])
    max_gt = np.array([pcd_ref[:,0].max(), pcd_ref[:,1].max(), pcd_ref[:,2].max()]) 
    dim_gt = max_gt - min_gt
    print("max dim", dim_gt.max(), max_gt, min_gt, flush=True)
    

    dms = []
    dm_names = []
    
    if sys.argv[1].endswith(".txt"):
        with open(sys.argv[1],"r") as fl:
            dms = fl.readlines()
            dm_names = dms[2::2]
            dms         = dms[1::2]
        for name, file in zip(dm_names, dms):
            name = name[:-1]
            file = file[:-1] # strip new line
    else: 
        dms = [sys.argv[1]]
        dm_names = [sys.argv[-1]]
            
    for name, file in zip(dm_names, dms):

        print(name, file)
        dm_opt = load_dm(file)
        

        
        scene = colmap.load_colmap_as_scene(sys.argv[3])
        view = scene.find_view_from_name(name)
        view.camera.scale_intrinsics(dm_opt.shape[1],dm_opt.shape[0])
        
        print("depth map --> wpos", flush=True)
        d_opt_vpos = positions_from_depth_map(dm_opt, view.camera.fx, view.camera.fy, view.camera.cx, view.camera.cy, view.camera.width, view.camera.height, 1, impl=np)
        d_opt_vpos = np.concatenate([d_opt_vpos, np.ones((d_opt_vpos.shape[0],1))], axis=-1)
        d_opt_wpos = d_opt_vpos @ view.C2W().T
        d_opt_wpos = d_opt_wpos[:,:3]
        
        # remove everything outside the bounding box
        # print(np.all(d_opt_wpos > min_gt[None,], axis=-1))
        # print(np.all(d_opt_wpos > min_gt[None,], axis=-1).squeeze().shape)


        print("remove everything outside the bounding box", flush=True)
        d_opt_wpos = d_opt_wpos[np.all(d_opt_wpos > min_gt[None,], axis=-1)]
        d_opt_wpos = d_opt_wpos[np.all(d_opt_wpos < max_gt[None,], axis=-1)]
        # print(d_opt_wpos.shape)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(d_opt_wpos)
        o3d.io.write_point_cloud("test.ply", pcd)

        
        print("compute accuracy", flush=True)
        print(name, accuracy(d_opt_wpos, pcd_tree), flush=True)
        
