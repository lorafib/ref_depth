import numpy as np
import torch
from pathlib import Path

from render import util

def calc_pcd_vs_refdm_diff(args, points2D_11, v_points3D, depth_image_ref):   
    ref_depths = util.tex_2d(depth_image_ref.get_absolute.unsqueeze(-1), points2D_11)
    valids = (ref_depths > args.cam_near_far[0]*2) # *2 for eps
    diff = torch.abs(ref_depths + v_points3D[...,2].view(-1,1))/ref_depths
    std_mean = torch.std_mean(diff[valids])
    print("PCD relative diff std, mean", std_mean[0].item(), std_mean[1].item())
    
    return diff


import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
def plot_pcd_diffs_on_img(img, points2D11, diff, filename=None):
    implot = plt.imshow(img)

    min_val = 0.
    max_val = 0.2
    col_diff = np.clip((diff - min_val)/(max_val-min_val), 0,1)

    col_diff = cmap(col_diff)
    points2D11[:,0] = (0.5*points2D11[:,0]+0.5)*img.shape[1]
    points2D11[:,1] = (0.5*points2D11[:,1]+0.5)*img.shape[0]
    scatter = plt.scatter(x=points2D11[:,0], y=points2D11[:,1], s=2.*np.ones_like(points2D11[:,1]), marker='o', c=col_diff )
    plt.axis("off")
    # Create a color bar to show the colormap
    norm = plt.Normalize(vmin=min_val*100, vmax=max_val*100)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), fraction=0.025, pad=0.01)
    cbar.set_label('Rel. Difference [\%]')
    
    
    # Show plot
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
                

@torch.no_grad()
def init_sparse_pcd_from_ref_dm(args, view, scene, depth_image_ref, perturb_std=0.02, num_points=5000, outlier_ratio=0.02):
    
    # create random 2d img locations for synthetic pcd
    points2D_11 = 2*torch.rand([num_points, 2])-1.
    
    # sample refernce depth map
    depths = util.tex_2d(depth_image_ref.get_absolute.unsqueeze(-1).cpu(), points2D_11)
    valids = (depths > args.cam_near_far[0]*2).squeeze()
    depths = depths[valids]
    points2D_11 = points2D_11[valids]
    num_points = depths.shape[0] # only use num valid points
    
    # add some noise
    perturbed_depths = depths*(1.+perturb_std*torch.randn([num_points, 1]))
    diff = (depths-perturbed_depths)/depths
    
    # add some outliers
    outlier_idx = (torch.rand([int(num_points*outlier_ratio)])*num_points).to(dtype=torch.int32)
    outliers = torch.rand([int(num_points*outlier_ratio),1])*(depths.max()-depths.min())+depths.min()
    perturbed_depths[outlier_idx] = outliers
    diff = torch.abs(depths-perturbed_depths)/depths
    
    # eval difference
    print("PCD relative diff std, mean", torch.std_mean(diff), torch.var_mean(diff))
    
    # get current intrinsics
    fx, fy, cx, cy, width, height, img_scale = util.scale_intrinsics(view.camera.fx, view.camera.fy, view.camera.cx, view.camera.cy, view.camera.width, view.camera.height, args.cam_resolution[1], args.cam_resolution[0])
    
    # to image space in px
    points2D = points2D_11.clone()
    points2D[:,0] = (points2D[:,0]*0.5+0.5)*width
    points2D[:,1] = (points2D[:,1]*0.5+0.5)*height
    
    # to view space ( inv K)
    v_points3D = torch.zeros(points2D.shape[0], 3)
    v_points3D[:,0] = (points2D[:,0] - cx)  * perturbed_depths.squeeze() / fx
    v_points3D[:,1] = (points2D[:,1] - cy)  * perturbed_depths.squeeze() / fy
    v_points3D[:,2] = -perturbed_depths.squeeze()                   # for gl convention
    v_points3D[:,1] = -v_points3D[:,1] # ??     # for gl convention
    
    # also sample color for visualization
    color_img = torch.tensor(util.load_image(Path(args.input_dir)/scene.color_path/view.image_name))
    col3D = util.tex_2d(color_img.cpu(), points2D_11)    
    
    # to device
    v_points3D  =  torch.Tensor(v_points3D).to(device="cuda", dtype=torch.float32).unsqueeze(0)
    points2D_11 =  torch.Tensor(points2D_11).to(device="cuda", dtype=torch.float32)
    
    return v_points3D, points2D_11, 255*col3D
    

@torch.no_grad()
def init_sparse_pcd(args, view, scene):

    from dataset.cv_dataset_utils.colmap.read_colmap_point_image_correspondences import get_points2D_and_3D_for_img, scatter_plot3D
    from dataset.cv_dataset_utils.conversion.external_utils import colmap_utils
    colmap_dir = Path(args.input_colmap_sparse) 

    # load colmap data including sparse pointcloud
    try:
        points3d    = colmap_utils.read_points3d_binary(colmap_dir/"points3D.bin")
        images      = colmap_utils.read_images_binary(colmap_dir/"images.bin")
    except:
        points3d    = colmap_utils.read_points3D_text(colmap_dir/"points3D.txt")
        images      = colmap_utils.read_images_text(colmap_dir/"images.txt")

    
    # use original image dimensions not scaled ones!
    width   = scene.views[0].camera.width
    height  = scene.views[0].camera.height
    
    # get correct img_id
    img_id = -1
    for colmap_image in images.values():
        if str(Path(view.image_name).stem) in colmap_image.name: img_id = colmap_image.id
    if img_id == -1: 
        print("img id was not found!") 
        exit()
    # get all 2d and 3d coordinates of all points visible in img_id
    points2D, points3D, col3D = get_points2D_and_3D_for_img(img_id, points3d, images, width, height)
    
    # points3D --> depth of refernce img via transfrom in view space (sign is handled in init of regularizer)
    v = view.W2C() 
    if args.cam_use_flip_mat: 
        v = util.cam_convention_transform(torch.Tensor(v), args.cam_use_flip_mat, use_rot_mat = False).numpy()
        print("this was not tested with this config!" )
        # exit()

    # to view space
    v_points3D = util.transform_pos(v, points3D)
    # is within near?
    depth = -v_points3D[:,:,2]
    print("depth", (depth).mean(), (depth).min(), (depth).max())
    valids = (depth > args.cam_near_far[0]*2)
    v_points3D = v_points3D[valids].unsqueeze(0)
    
    # points2D --> -1,1 for grid sample
    points2D =  torch.Tensor(points2D).to(device="cuda", dtype=torch.float32)
    print("Num points before invalid filtering:", points2D.shape[0])
    points2D = points2D[valids.squeeze(0)]
    print("Num points after  invalid filtering:", points2D.shape[0])

    points2D_11 = points2D
    points2D_11[:,0] = 2.*(points2D_11[:,0] / width)  -1.
    points2D_11[:,1] = 2.*(points2D_11[:,1] / height) -1.
    
    return v_points3D, points2D_11, col3D
 
 
 