import open3d as o3d
from PIL import Image
import numpy as np
import os
import tqdm
import cv2
import pandas as pd

import sys
try:
    sys.path.append("..")
    from conversion.scene import *
    import conversion.colmap  as colmap
except:
    from ..conversion.scene import *
    from ..conversion import colmap

join = os.path.join
basename = os.path.basename



def read_color_scannet(fn):
    image = cv2.imread(str(fn))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_depth_scannet(fn):
    return np.array(Image.open(fn), dtype='u2')

def to_rgbd(color, depth, depth_scale, depth_trunc):
    if color.shape[:2] != depth.shape[:2]:
        color = cv2.resize(color, (depth.shape[1], depth.shape[0]))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color), o3d.geometry.Image(depth),
                                                      convert_rgb_to_intensity=False, depth_scale=depth_scale, depth_trunc=depth_trunc)
    return rgbd


def write_as_ply(path, pcd):
    xyz = np.array(pcd.points, np.float32)
    normals = np.array(pcd.normals, np.float32)
    colors = np.array(pcd.colors, np.float32)

    if np.array(pcd.colors).dtype == np.uint8:
        colors /= 255.

    df = {
        'x': xyz[:,0],
        'y': xyz[:,1],
        'z': xyz[:,2],
        'nx': normals[:,0],
        'ny': normals[:,1],
        'nz': normals[:,2],
        'red': colors[:,0],
        'green': colors[:,1],
        'blue': colors[:,2],
    }

    df = pd.DataFrame(df, columns=[
        'x','y','z',
        'nx','ny','nz',
        'red','green','blue'])

    write_ply(path, points=df, as_text=False)


def jitter(pcd, mag):
    pcd = o3d.geometry.PointCloud(pcd)
    pts = np.array(pcd.points)
    pts_j = pts + np.random.rand(*pts.shape) * mag - 0.5 * mag
    pcd.points = o3d.utility.Vector3dVector(pts_j)
    return pcd

def build_pointcloud_from_scene(scene, args):
    more_args = {}
    more_args["n_frames"] = len(scene.views)
    more_args["use_eigvecs"] = False
    more_args["base_vs"] = 0.001
    more_args["down_sample"] = True
    more_args["remove_outliers"] = True
    
    # if more_args["n_frames"] > 300: args.take_each = 2
    
    pcd_combined = o3d.geometry.PointCloud()
    print('building point cloud...')
    for i, view in tqdm.tqdm(scene.views.items()):
        pose = view.C2W()
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width=view.camera.width, height=view.camera.height, 
                                                       fx=view.camera.fx, fy=view.camera.fy, 
                                                       cx=view.camera.cx, cy=view.camera.cy, )

        if not np.isfinite(pose).all():
            print('skip ', i)
            continue

        if i % args.take_each != 0:
            continue

        color_img = read_color_scannet(Path(args.color_dir)/(view.image_name))
        depth_img = read_depth_scannet(Path(args.depth_dir)/(view.image_name))
        
        rgbd = to_rgbd(color_img, depth_img, args.depth_scale, args.depth_trunc)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

        if more_args["remove_outliers"]:
            pcd, _ = o3d.geometry.PointCloud.remove_statistical_outlier(pcd, nb_neighbors=int(30), std_ratio=float(1.25))
     
        pcd.transform(pose)
        pcd_combined += pcd
        


    print('COMBINED')
    print(pcd_combined)
    
    voxel_size = args.voxel_size 
    # for 1m extension a point evey 1cm seems appropriate to have good sampling rate 
    # --> scale by largest eig val to keep relative sampling (very approximately)
    draw_geoms = []
    vscale = 30  
    if more_args["down_sample"] and args.voxel_size == 0.0:
        try:
            if more_args["use_eigvecs"]:
                m, cov =  o3d.geometry.PointCloud.compute_mean_and_covariance(pcd_combined)
                eigval, eigvec = np.linalg.eig(cov)
                # print(eigval)
                if args.gui:
                    box = o3d.geometry.OrientedBoundingBox(center=m, R=eigvec, extent=eigval)
                    draw_geoms.append(box)
                vscale = eigval.max()
            else:
                bbox =  pcd_combined.get_oriented_bounding_box()
                if args.gui:
                    draw_geoms.append(bbox)
                # print(np.asarray(bbox.get_box_points()), bbox.extent)
                vscale = bbox.extent.max()
        except: pass   
        voxel_size = more_args["base_vs"] * vscale 
    elif more_args["down_sample"] and args.voxel_size != 0.0:
        voxel_size = args.voxel_size # more_args["base_vs"] * vscale      
    
    more_args["actual_voxel_size"] = voxel_size     

    if more_args["down_sample"]:
        print('downsampling... with voxel_size',more_args["base_vs"],"*", vscale, "=", voxel_size)    
        pcd_combined = o3d.geometry.PointCloud.voxel_down_sample(pcd_combined, voxel_size)
        print(pcd_combined)

    if more_args["remove_outliers"]:
        o3d.geometry.PointCloud.remove_statistical_outlier(pcd_combined, nb_neighbors=100, std_ratio=1.5)

    # print('calculating normals...')
    # o3d.geometry.PointCloud.estimate_normals(pcd_combined)

    if args.jitter:
        # apply small jitter to points to destroy grid pattern, which *probably* could lead to model overfitting
        pcd_combined = jitter(pcd_combined, args.voxel_size)

    # os.makedirs(join(args.input, 'geometry'), exist_ok=True)

    # get_name = lambda ext: 'pointcloud_te_{}_vs_{}{}.{}'.format(args.take_each, args.voxel_size, '_jit' if args.jitter else '', ext)

    # # name_pcd = join(args.input, 'geometry', get_name('pcd'))
    # # print(name_pcd)
    
    draw_geoms.append(pcd_combined)
    if args.gui:
        o3d.visualization.draw_geometries(draw_geoms)
    
    
    o3d.io.write_point_cloud(str(args.output_name), pcd_combined)

    with open(str(args.output_name)+"_args.txt","w") as settings_file:
        settings_file.write(str(args))
        settings_file.write(str(more_args))
            
    # name_ply = join(args.input, 'geometry', get_name('ply'))
    # print(name_ply)
    # write_as_ply(name_ply, pcd_combined)


if __name__ == '__main__':
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--input_format', type=str, required=True)
    parser.add_argument('--color_dir', type=str, required=True)
    parser.add_argument('--depth_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    
    parser.add_argument('--take-each', type=int, default=1)
    parser.add_argument('--voxel-size', type=float, default=0.3)
    parser.add_argument('--depth-scale', type=float, default=1000.0)
    parser.add_argument('--depth-trunc', type=float, default=8.0)
    parser.add_argument('--no-jitter', action='store_true')     
    parser.add_argument('--gui', action='store_true')    

    args = parser.parse_args()

    args.jitter = not args.no_jitter
    
    print(args)

    scene = colmap.load_colmap_as_scene(args.input_dir)
    
    build_pointcloud_from_scene(scene, args)
    