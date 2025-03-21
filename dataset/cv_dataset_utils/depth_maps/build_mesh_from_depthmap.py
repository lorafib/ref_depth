import glob
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
from PIL import Image
import numpy as np
import os
import cv2

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
    print("load color", fn)
    image = cv2.imread(str(fn))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_depth_scannet(fn):
    print("load depth", fn)
    return np.array(Image.open(fn), dtype='u2')

def to_o3d_rgbd(color, depth, depth_scale, depth_trunc):
    if color.shape[:2] != depth.shape[:2]:
        color = cv2.resize(color, (depth.shape[1], depth.shape[0]))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color), o3d.geometry.Image(depth),
                                                      convert_rgb_to_intensity=False, depth_scale=depth_scale, depth_trunc=depth_trunc)
    return rgbd


def are_vaild_points(pos):
    valid_depth = (np.all(np.abs(pos[:,2]) < 100))
    valid_depth = valid_depth and (np.all(np.abs(pos[:,2]) > 0.001))
    valid_values = (not np.any(np.isnan(pos))) and (not np.any(np.isinf(pos)))
    return valid_depth and valid_values

def depth_to_mesh(depth_image, fx, fy, cx, cy, depth_scale=1.0, mesh_downscale=1):
    """
    Convert an RGB-D image to a mesh.

    Parameters:
    - depth_image: A 2D NumPy array with depth values.
    - fx, fy: Focal lengths of the camera in pixels.
    - cx, cy: Optical center of the camera in pixels.
    - depth_scale: Scale factor to convert depth values to meters.

    Returns:
    - points: A 2D NumPy array (Nx3) with XYZ coordinates of points.
    - triangles: A 2D NumPy array (Nx3) with vertex indices forming triangles.
    - uvs: A 2D NumPy array (Nx2) with texture coordinates for the points.
    """
    img_height, img_width = depth_image.shape
    tri_height, tri_width = depth_image[::mesh_downscale, ::mesh_downscale].shape
    xx, yy = np.meshgrid(np.arange(tri_width), np.arange(tri_height))
    xx *= mesh_downscale
    yy *= mesh_downscale

    # Convert depth image to 3D points
    Z = depth_image * depth_scale
    Z = Z[::mesh_downscale, ::mesh_downscale]
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy
    Z = -Z # for gl convention
    Y = -Y # for gl convention
    
    # Flatten the X, Y, Z coordinates into a (N, 3) array
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    

    # Loop through each pixel except for the last row and column
    triangles = []
    for y in range(tri_height - 1):
        for x in range(tri_width - 1):
            idx = y * tri_width + x
            
            # Triangle 1
            idx1 = [idx, idx + 1, idx + tri_width]     
            if are_vaild_points(points[idx1]):
                triangles.append(idx1)
            else:
                triangles.append([-1,-1,-1])
                
            # Triangle 2
            idx2 = [idx + 1, idx + 1 + tri_width, idx + tri_width]
            if are_vaild_points(points[idx2]):
                triangles.append(idx2)
            else:
                triangles.append([-1,-1,-1])
    triangles = np.array(triangles)
            
    xx, yy = np.meshgrid(np.arange(tri_width), np.arange(tri_height))
    xx = xx/float(tri_width) +0.5/float(img_width)
    yy = yy/float(tri_height) +0.5/float(img_height)
    uvs = np.vstack((xx.flatten(), yy.flatten())).T
                        
    return points, triangles, uvs

def np_from_o3d_mesh(mesh):
    return  np.asarray(mesh.vertices),\
            np.asarray(mesh.triangles),\
            np.asarray(mesh.triangle_uvs)

def create_o3d_mesh(vertices, triangles, tcs=None, rgb_image=None):
    # Create a mesh from the vertices and triangles
    mesh = o3d.geometry.TriangleMesh()
    # print("mesh")
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # print("verts")

    # print(triangles.shape, type(triangles), triangles.dtype)
    triangles = np.array(triangles, dtype=np.int32)
    # print(triangles.shape, type(triangles), triangles.dtype)
    triangles = o3d.utility.Vector3iVector(triangles)
    # print(triangles)
    mesh.triangles = triangles
    # print("tris")

    if type(tcs) != type(None):
        # print(tcs.shape, type(tcs), tcs.dtype)
        tcs = np.array(tcs, dtype=np.float32)
        # print(tcs.shape, type(tcs), tcs.dtype)
        tcs = o3d.utility.Vector2dVector(tcs)
        # print(tcs)

        mesh.triangle_uvs = o3d.utility.Vector2dVector(tcs)
        # print("uvs")
    
    if type(rgb_image) != type(None):
        mesh.textures = [o3d.geometry.Image(rgb_image)] 
        # print("tx")
    # Compute normals
    mesh.compute_vertex_normals()
    # print("norms")
    return mesh

def o3d_rgbd_to_o3d_mesh(rgbd_image, camera_intrinsics):
    """
    Convert an RGBD image to a mesh by triangulating points from neighboring pixels.

    Parameters:
    - rgbd_image: An Open3D RGBDImage object.
    - camera_intrinsics: The camera intrinsics necessary for the point cloud projection.

    Returns:
    - mesh: An Open3D TriangleMesh object.
    """
    # Create a point cloud from the RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics, project_valid_depth_only=False)

    # # Estimate normals
    # pcd.estimate_normals()

    # Triangulate points from neighboring pixels
    vertices = np.asarray(pcd.points)
    num_rows, num_cols = camera_intrinsics.height, camera_intrinsics.width
    print(vertices.shape, num_rows*num_cols)
    triangles = []
    tcs = []

    # Loop through each pixel except for the last row and column
    use_every_nth=2
    for y in range(num_rows - 1):
        for x in range(num_cols - 1):
            idx = y * num_cols + x
            # Triangle 1
            triangles.append([idx, idx + 1, idx + num_cols])
            # Triangle 2
            triangles.append([idx + 1, idx + 1 + num_cols, idx + num_cols])
            
            # Texture coordinates per triangle/// why is this so ugly?
            # Triangle 1
            tcs.append([(x+0.5)/num_cols,   (y+0.5)/num_rows])
            tcs.append([(x+0.5)/num_cols,   (y+1+0.5)/num_rows])
            tcs.append([(x+1+0.5)/num_cols, (y+0.5)/num_rows])
            # Triangle 2           
            tcs.append([(x+0.5)/num_cols,   (y+1+0.5)/num_rows])
            tcs.append([(x+1+0.5)/num_cols, (y+1+0.5)/num_rows])
            tcs.append([(x+1+0.5)/num_cols, (y+0.5)/num_rows])
            
            
    # filter out invalids
    vertices[np.isnan(vertices)] = 0.0
            

    # Create a mesh from the vertices and triangles
    mesh = create_o3d_mesh(vertices, triangles, tcs, rgbd_image.color)
    return mesh, pcd

# def write_as_ply(path, pcd):
#     xyz = np.array(pcd.points, np.float32)
#     normals = np.array(pcd.normals, np.float32)
#     colors = np.array(pcd.colors, np.float32)

#     if np.array(pcd.colors).dtype == np.uint8:
#         colors /= 255.

#     df = {
#         'x': xyz[:,0],
#         'y': xyz[:,1],
#         'z': xyz[:,2],
#         'nx': normals[:,0],
#         'ny': normals[:,1],
#         'nz': normals[:,2],
#         'red': colors[:,0],
#         'green': colors[:,1],
#         'blue': colors[:,2],
#     }

#     df = pd.DataFrame(df, columns=[
#         'x','y','z',
#         'nx','ny','nz',
#         'red','green','blue'])

#     write_ply(path, points=df, as_text=False)


def jitter(pcd, mag):
    pcd = o3d.geometry.PointCloud(pcd)
    pts = np.array(pcd.points)
    pts_j = pts + np.random.rand(*pts.shape) * mag - 0.5 * mag
    pcd.points = o3d.utility.Vector3dVector(pts_j)
    return pcd


more_args = {}
more_args["use_eigvecs"] = False
more_args["base_vs"] = 0.001
more_args["down_sample"] = True
more_args["remove_outliers"] = True
def build_mesh_from_scene(scene, args):
    more_args["n_frames"] = len(scene.views)
    
    pcd_combined = o3d.geometry.PointCloud()
    print('building point cloud...')
    
    view = scene.find_view_from_name(args.view_name)
    pose = view.C2W()

    color_img = read_color_scannet(Path(args.color_dir)/(view.image_name))
    depth_img = read_depth_scannet(Path(args.depth_dir)/(view.image_name))
    
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width=view.camera.width, height=view.camera.height, 
                                                    fx=view.camera.fx, fy=view.camera.fy, 
                                                    cx=view.camera.cx, cy=view.camera.cy, )
    rgbd = to_o3d_rgbd(color_img, depth_img, args.depth_scale, args.depth_trunc)
    
    # Generate mesh
    use_o3d_parser = False
    if use_o3d_parser: 
        mesh,pcd = o3d_rgbd_to_o3d_mesh(rgbd, intrinsics)
    else:
            # Create a point cloud from the RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, project_valid_depth_only=False)
        depth_img = depth_img.astype(np.float64)
        depth_img = depth_img/depth_img.max()
        depth_img = depth_img *3 +2.0
        print(depth_img.max())
        vertices, triangles, uvs = depth_to_mesh(depth_img, fx=view.camera.fx, fy=view.camera.fy, 
                                                    cx=view.camera.cx, cy=view.camera.cy, depth_scale=1.0, mesh_downscale=args.skip_pixel)        
        # Create a mesh from the vertices and triangles
        mesh = create_o3d_mesh(vertices, triangles, uvs, color_img)
    mesh.transform(pose) 
        
    return mesh, pcd

def simplify_mesh(o3d_mesh, simplification_ratio):
    num_tris = int(simplification_ratio*len(o3d_mesh.triangles))
    print("simplify mesh from:", len(o3d_mesh.triangles), "to:", num_tris)
    print("numv_uvs mesh from:", len(o3d_mesh.triangle_uvs))
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=num_tris, boundary_weight=100.0) #1e12, maximum_error=(1e12*(1/1080)*(1/1920)*0.5)-1000.)
    print("numv_uvs mesh to:", len(o3d_mesh.triangle_uvs))
    return o3d_mesh

if __name__ == '__main__':
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--color_dir', type=str, required=True)
    parser.add_argument('--depth_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument("--view_name", type=str, required=True)
    
    parser.add_argument('--voxel-size', type=float, default=0.3)
    parser.add_argument('--depth-scale', type=float, default=1000.0)
    parser.add_argument('--depth-trunc', type=float, default=8.0)
    parser.add_argument('--gui', action='store_true')    
    parser.add_argument('--skip_pixel', type=int, default=1)

    parser.add_argument('--mesh_simplification_ratio', type=float, default=-1.0)
    args = parser.parse_args()
    
    print(args)

    if args.input_dir == "dummy":
        scene = BaseScene()
        width, height = 1296, 968
        cam = BaseCamera(0, width, width, 0.5*width, 0.5*height, width, height)
        view = BaseView(0, np.eye(3,3), np.zeros((3)), cam, "./"+args.view_name)
        scene.cameras[args.view_name] = cam
        scene.views[args.view_name] = view
    else:
        scene = colmap.load_colmap_as_scene(args.input_dir)
    
    o3d_mesh, pcd = build_mesh_from_scene(scene, args)
    
    if args.mesh_simplification_ratio > 0.0:
        if args.mesh_simplification_ratio > 1.0:
            print("mesh simplification ratio invalid. is bigger than 1!")
        else:
            o3d_mesh = simplify_mesh(o3d_mesh, args.mesh_simplification_ratio)
    # if args.gui:
    #     o3d.visualization.draw_geometries(o3d_mesh)
    
    print("write mesh")
    o3d.io.write_triangle_mesh(filename=str(args.output_name), mesh=o3d_mesh)
    o3d.io.write_point_cloud(str(args.output_name)+".ply", pcd)

    with open(str(args.output_name)+"_args.txt","w") as settings_file:
        settings_file.write(str(args))
        settings_file.write(str(more_args))
            
    # name_ply = join(args.input, 'geometry', get_name('ply'))
    # print(name_ply)
    # write_as_ply(name_ply, pcd_combined)