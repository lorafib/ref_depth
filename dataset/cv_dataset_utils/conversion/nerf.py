from pathlib import Path
import json

try:
    from  scene import *
    import external_utils.math_utils as math_utils
    import external_utils.nerf_utils as nerf_utils
except:
    from .scene import *
    from .external_utils import math_utils
    from .external_utils import nerf_utils

def nerf_to_camera(t_file):
    camera = BaseCamera( 0, t_file["fl_x"], t_file["fl_y"], 
                            t_file["cx"], t_file["cy"], 
                            t_file["w"], t_file["h"])
    return camera

    

def load_nerf_as_scene(input_dir, transform_file="transforms.json") -> BaseScene:
    scene = BaseScene()
    input_dir = Path(input_dir)
    
    t_file = json.load(open(input_dir/transform_file, "r"))
    
    camera = nerf_to_camera(t_file)
    scene.cameras = {camera.id: camera}
    scene.color_path = Path(t_file["frames"][0]["file_path"]).parent # WARNING: assumes only one path, if there are different one it'll break
    
    num_views = len(t_file["frames"])
    for i in range(num_views):
        image_name = Path(t_file["frames"][i]["file_path"]).name
        transform = np.array(t_file['frames'][i]['transform_matrix'])
        transform = nerf_utils.C2W_from_nerf(transform) # TODO to make consistent with colmap, matrix would need to be removed
        scene.views[i] = view_from_C2W(i, transform, camera, image_name)
        

    return scene
    
    

def camera_to_nerf(camera: BaseCamera, aabb_scale):
    print("Warning: only pinhole cameras are exported!")
    angle_x = math_utils.fov_from_f_radians(camera.fx, camera.width)
    angle_y = math_utils.fov_from_f_radians(camera.fy, camera.height)
    nerf_cam = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": camera.fx,
        "fl_y": camera.fy,
        "k1": 0.,
        "k2": 0.,
        "k3": 0.,
        "k4": 0.,
        "p1": 0.,
        "p2": 0.,
        "is_fisheye": False,
        "cx": camera.cx,
        "cy": camera.cy,
        "w": camera.width,
        "h": camera.height,
        "aabb_scale": aabb_scale,
    }
    return nerf_cam
    
def write_scene_as_nerf(scene : BaseScene, output_dir, rescale=True, aabb_scale=64):
    # camera
    key = list(scene.cameras.keys())[0]
    if len(scene.cameras) > 1:
        print(f"Waring: nerf format only supports one camera. Used camera is index {key}.")
    out = camera_to_nerf(scene.cameras[key], aabb_scale=aabb_scale)
    
    # views
    print("Waring: sharpness is just a placeholder value (1.0)!")
    
    nerf_c2ws = {k: v.C2W() for k, v in scene.views.items() }
    if rescale:
        nerf_c2ws = nerf_utils.rescaled_nerf_from_C2Ws(nerf_c2ws)
    else:
        nerf_c2ws = nerf_utils.nerf_from_C2Ws(nerf_c2ws)
        
    nerf_views = []
    for v in scene.views.values():
        nerf_views.append({"file_path":str(scene.color_path/v.image_name),"sharpness":1.0,
                           "transform_matrix": nerf_c2ws[v.id].tolist()})
    out["frames"] = nerf_views
    
    # write out
    output_filename = Path(output_dir)/"transforms.json"
    with open(output_filename, "w") as output_file:
        print("write", output_filename)
        json.dump(out, output_file, indent=2)
