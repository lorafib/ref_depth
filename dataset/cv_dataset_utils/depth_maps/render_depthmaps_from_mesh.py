import open3d as o3d
from PIL import Image
import numpy as np
import os
import tqdm
import cv2
import pyrender
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Options include 'egl', 'osmesa'

import sys
sys.path.append("./depth_maps")
import dmio
try:
    sys.path.append("..")
    from conversion.scene import *
    import conversion.colmap  as colmap
    import conversion.nerf  as nerf
except:
    from ..conversion.scene import *
    from ..conversion import colmap
    from ..conversion import nerf

def load_pyrender_mesh(mesh_path):
    
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.vertex_colors).astype(np.float32).shape[0])
    mesh.paint_uniform_color((0.7, 0.7, 0.7))

    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    colors = np.asarray(mesh.vertex_colors).astype(np.float32)
    normals = np.asarray(mesh.vertex_normals).astype(np.float32)
    
    mesh = pyrender.Mesh(
            primitives=[
                pyrender.Primitive(
                    positions=verts,
                    normals=normals,
                    color_0=colors,
                    indices=faces,
                    mode=pyrender.GLTF.TRIANGLES,
                )
            ],
            is_visible=True,
        )
    return mesh



def render_depth_map(
    # renderer,
    o3d_scene,
    view,
):

    camera = o3d.camera.PinholeCameraParameters()
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width=view.camera.width, height=view.camera.height, 
                                    fx=view.camera.fx, fy=view.camera.fy, 
                                    cx=view.camera.cx, cy=view.camera.cy, )
    camera.set_intrinsics(intrinsics)
    camera.extrinsic(view.C2W())

    # Render the depth map
    depth_map = o3d_scene.render_depth_image(camera)

    return depth_map


if __name__ == "__main__":
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--input_mesh', type=str, required=True)
    parser.add_argument('--output_depth_dir', type=str, required=True)
    parser.add_argument('--output_color_dir', type=str, required=True)
    parser.add_argument('--max_imgs', type=int, default=0)
    
    args = parser.parse_args()
    print(args)

    # try:
    #     scene = colmap.load_colmap_as_scene(args.input_dir)
    #     print("WARNING for TNT: behavior might have changed!!")
    # except:
    scene = nerf.load_nerf_as_scene(args.input_dir, transform_file="transforms_resized_undistorted.json")
    scene_test = colmap.load_colmap_as_scene(args.input_dir+"../colmap")
    colmap.write_scene_as_colmap(scene, args.input_dir)
    # before:
    # scene.views = {int(str(Path(v.image_name).stem)): v for v in scene.views.values()}
    scene.views = {str(Path(v.image_name).stem): v for v in scene.views.values()}
    # print(scene.views)
    
    # init mesh
    print("load mesh")
    mesh = load_pyrender_mesh(str(args.input_mesh))
    mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
    
    camera = scene.cameras[list(scene.cameras.keys())[0]]
    # renderer must be initialized outside of loop!
    renderer = pyrender.OffscreenRenderer(camera.width, camera.height)

    znear=0.05
    zfar=99
    
    print(f"render depth maps to {args.output_depth_dir}")
    Path(args.output_depth_dir).mkdir(parents=True, exist_ok=True)
    print(f"render color imgs to {args.output_color_dir}")
    Path(args.output_color_dir).mkdir(parents=True, exist_ok=True)

    # render all views
    for i,v in enumerate(sorted(list(scene.views.values()), key=lambda x: x.image_name)):
        
        v_test= scene_test.find_view_from_name(v.image_name)
        
        if args.max_imgs > 0 and i >= args.max_imgs: break
        # init scene
        pyscene = pyrender.Scene(ambient_light=[1., 1., 1.])

        
        pyscene.add_node(mesh_node)
        cam = pyrender.IntrinsicsCamera(
            fx=v.camera.fx,
            fy=v.camera.fy,
            cx=v.camera.cx,
            cy=v.camera.cy,
            znear=znear,
            zfar=zfar,
        )
        T = v.C2W()
        T = v_test.C2W() ### eewwwwww!!!! FIXME
        cv2gl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        # T = T @  np.linalg.inv(cv2gl)
        # T = np.linalg.inv(T)
        T = T @ cv2gl

        cam_node = pyrender.Node(camera=cam, matrix=T)
        pyscene.add_node(cam_node)


        color, depth = renderer.render(pyscene, flags=pyrender.constants.RenderFlags.FLAT)
        
        dmio.write_depth_image((Path(args.output_depth_dir)/v.image_name).with_suffix(".png"), depth)
        Image.fromarray((255*color).astype(np.uint8)).save((Path(args.output_color_dir)/v.image_name))

    