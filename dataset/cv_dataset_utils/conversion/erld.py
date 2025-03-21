from pathlib import Path


try:
    from  scene import *
    import external_utils.math_utils as math_utils
    import external_utils.erld_utils as erld_utils
except:
    from .scene import *
    from .external_utils import math_utils
    from .external_utils import erld_utils

def load_oldlivenvs_as_scene(input_dir, color_dir="color", depth_dir="depth", traj_filename="KeyframeTrajectory.txt") -> BaseScene:
    input_dir = Path(input_dir)
    dscene = DepthMapScene()
    dscene.color_path = input_dir / color_dir
    dscene.depth_path = input_dir / depth_dir
    
    associations = erld_utils.load_associations(Path(input_dir) / "associations.txt")
    trajectory = erld_utils.load_trajectory(Path(input_dir) / traj_filename)
    
    # load cameras        
    color_K = erld_utils.load_K(input_dir / "intrinsic"/ "intrinsic_color.txt" ) 
    depth_K = erld_utils.load_K(input_dir / "intrinsic"/ "intrinsic_depth.txt") 
    color_width, color_height = erld_utils.load_resolution(input_dir / "intrinsic"/ "resolution.txt") 
    depth_width, depth_height = erld_utils.load_resolution(input_dir / "intrinsic"/ "resolution.txt" ) 
        
    color_camera = BaseCamera(0, 
                              color_K[0,0], color_K[1,1], color_K[0,2], color_K[1,2], 
                              color_width, color_height)
    depth_camera = BaseCamera(0, 
                              depth_K[0,0], depth_K[1,1], depth_K[0,2], depth_K[1,2], 
                              depth_width, depth_height)
    dscene.cameras = {0: color_camera}
    dscene.depth_cameras = {0: depth_camera}
    
    # load views
    id_to_time = {}
    for i, t in enumerate(trajectory):
        T = t["T"]
        R = math_utils.rotm_from_quat(t["q"])
        R, T = math_utils.inv_R_T(R,T)
        # timestamp precision exported by slam and by zed camera sdk are different
        # --> round  
        time_str = f"{t['timestamp']:.05f}"
            
        dscene.depth_views[i] = BaseView(i, R, T, depth_camera, associations[time_str])
        dscene.views[i] = BaseView(i, R, T, color_camera, associations[time_str])
        id_to_time[i] = float(time_str)
        
    return dscene, id_to_time
    

def load_erld_as_scene(input_dir, color_dir="color", depth_dir="depth", traj_filename="keyframe_trajectory.txt") -> BaseScene:
    input_dir = Path(input_dir)
    dscene = DepthMapScene()
    dscene.color_path = input_dir / color_dir
    dscene.depth_path = input_dir / depth_dir
    
    associations = erld_utils.load_associations(Path(input_dir) / "camera_tracking" / "associations.txt")
    trajectory = erld_utils.load_trajectory(Path(input_dir) / "camera_tracking" / traj_filename)
    
    # load cameras        
    color_K = erld_utils.load_K(input_dir / "camera_calibration" / "intrinsic_color.txt" ) 
    depth_K = erld_utils.load_K(input_dir / "camera_calibration" / "intrinsic_depth.txt") 
    color_width, color_height = erld_utils.load_resolution(input_dir / "camera_calibration" / "resolution_color.txt") 
    depth_width, depth_height = erld_utils.load_resolution(input_dir / "camera_calibration" / "resolution_depth.txt" ) 
        
    color_camera = BaseCamera(0, 
                              color_K[0,0], color_K[1,1], color_K[0,2], color_K[1,2], 
                              color_width, color_height)
    depth_camera = BaseCamera(0, 
                              depth_K[0,0], depth_K[1,1], depth_K[0,2], depth_K[1,2], 
                              depth_width, depth_height)
    dscene.cameras = {0: color_camera}
    dscene.depth_cameras = {0: depth_camera}
    
    # load views
    id_to_time = {}
    for i, t in enumerate(trajectory):
        T = t["T"]
        R = math_utils.rotm_from_quat(t["q"])
        R, T = math_utils.inv_R_T(R,T)
        # timestamp precision exported by slam and by zed camera sdk are different
        # --> round  
        time_str = f"{t['timestamp']:.05f}"
            
        dscene.depth_views[i] = BaseView(i, R, T, depth_camera, associations[time_str])
        dscene.views[i] = BaseView(i, R, T, color_camera, associations[time_str])
        id_to_time[i] = float(time_str)
        
    return dscene, id_to_time
    

    

# def write_scene_as_erld(scene : BaseScene, output_dir, format=".txt"):
    