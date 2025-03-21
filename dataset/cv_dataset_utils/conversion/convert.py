import configargparse

from scene import *
import colmap
import erld
import nerf
import geomvsnet
import os

def guess_scene_type(input_dir):
    pass
    


def convert(args):
    
    # load scene
    scene = None
    if args.input_format == "colmap":
        scene = colmap.load_colmap_as_scene(args.input_dir)
    if args.input_format == "erld":
        traj_file = "camera_trajectory.txt" if "full_traj" in args.optional else "keyframe_trajectory.txt"
        scene, id_to_timestamp = erld.load_erld_as_scene(args.input_dir, traj_filename=traj_file)
        #TODO handle checks for intrinsics
    if args.input_format == "livenvs":
        traj_file = "CameraTrajectory.txt" if "full_traj" in args.optional else "KeyframeTrajectory.txt"
        scene, id_to_timestamp = erld.load_oldlivenvs_as_scene(args.input_dir, traj_filename=traj_file)
        #TODO handle checks for intrinsics
        
    if scene == None:
        print("error: invalid input format")
        
    if ".png" in args.optional:
        scene.replace_image_extension(".png")
    
    # --------------------------------------------------
    # -- formats that cannot handle depth 
    # -- they will only consider color information aka members of BaseScene
    if type(scene) == DepthMapScene:
        print("warning: data loss may occur!", 
              "dataset includes depth information,",
              "your chosen export format will only consider color information.")
    # write scene
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_format == "colmap":
        colmap.write_scene_as_colmap(scene, args.output_dir, 
                                     ".bin" if ".bin" in args.optional else ".txt") 
    if args.output_format == "nerf":
        aabb_scale = 32 
        if "aabb_scale=64" in args.optional: aabb_scale = 64
        if "aabb_scale=16" in args.optional: aabb_scale = 16
        nerf.write_scene_as_nerf(scene, args.output_dir, rescale=True, aabb_scale=aabb_scale)   
        
    if args.output_format == "geomvsnet":
        # for now assume some values
        depth_args = {}
        depth_args["min"] = args.geomvsnet_dmin
        depth_args["max"] = args.geomvsnet_dmax
        depth_args["interval"] = args.geomvsnet_dinterval
        depth_args["steps"] = int((depth_args["max"] - depth_args["min"]) / depth_args["interval"])

        
        geomvsnet.write_scene_as_geomvsnet(scene, args.output_dir, args.geomvsnet_num_nbs, depth_args, args.geomvsnet_step_size)
        




if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--input_format",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_format",
        type=str,
        required=True
    )
    parser.add_argument(
        "--optional",
        nargs="+",
        type=str,
        required=False
    )
    
    
    
    parser.add_argument(
        "--geomvsnet_step_size",
        type=str,
        default="1"
    )
    parser.add_argument(
        "--geomvsnet_num_nbs",
        type=int,
        default=10
    )
    parser.add_argument(
        "--geomvsnet_dmin",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--geomvsnet_dmax",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--geomvsnet_dinterval",
        type=float,
        default=1e-2 # we assume mm
    )
    
    
    args = parser.parse_args()
    if args.optional == None: args.optional = []
    
    print(args)
    convert(args)
    