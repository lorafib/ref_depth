import numpy as np
import configargparse
import os

import colmap as colmap
import nerf as nerf
from scene import *


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
        default="colmap"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--view_list",
        type=str,
        nargs='+',
        required=True
    )
    
     
    args = parser.parse_args()
    # if args.optional == None: args.optional = []
       
    print(args)
    
    
    # load scene
    if args.input_format == "colmap":
        scene_complete = colmap.load_colmap_as_scene(args.input_dir)
    elif args.input_format == "nerf":
        scene_complete = nerf.load_nerf_as_scene(args.input_dir)
        
    print(scene_complete.cameras)
        
    # create name dict
    views =    dict(sorted({str(Path(v.image_name).stem): v for v in scene_complete.views.values()}.items()))
    
    # select views
    selected_views = {v.id : v for k,v in views.items() if k in args.view_list}
    
    # write out
    scene_selected = scene_complete
    scene_selected.views = selected_views
    os.makedirs(args.output_dir, exist_ok=True)
    if args.input_format == "colmap":
        colmap.write_scene_as_colmap(scene_selected, args.output_dir)
    elif args.input_format == "nerf":
        nerf.write_scene_as_nerf(scene_selected, args.output_dir, rescale=False)
    
    