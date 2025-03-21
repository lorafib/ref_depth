from pathlib import Path

try:
    from  scene import *
    import external_utils.geomvsnet_utils as geomvsnet_utils
except:
    from .scene import *
    from .external_utils import geomvsnet_utils
    

def load_geomvsnet_as_scene(input_dir) -> BaseScene:
    scene = BaseScene()
    input_dir = Path(input_dir)
    
    return NotImplementedError


def write_scene_as_geomvsnet(scene : BaseScene, output_dir, num_nbs, depth_args : dict, step_size):
    output_dir = Path(output_dir)
    
    # write view files
    cams_dir = output_dir/"cams"
    cams_dir.mkdir(parents=True, exist_ok=True)
    for view in scene.views.values():
        # filename = cams_dir/f"{view.id:04d}_cam.txt"
        filename = cams_dir/f"{Path(view.image_name).stem}_cam.txt"
        geomvsnet_utils.write_view_file(filename, view.W2C(), view.camera.K(),
                                        depth_args["min"], depth_args["max"], depth_args["interval"], depth_args["steps"])
    
    # estimate neigbors
    if step_size == "max":
        step_size = len(scene.views)//num_nbs # distribute as far a possible
    else:
        step_size = int(step_size)

    # write pair.txt
    view_list = list(scene.views.values())
    print(view_list)
    if num_nbs > len(view_list):
        num_nbs = len(view_list)
        print("WARNING: numb_nbs capped to number of total views")
    with open(output_dir/"pair.txt", "w") as pair_file:
        pair_file.write(f"{len(view_list)}\n")
        for i, view in enumerate(view_list):
            start = i-(step_size*num_nbs)//2
            if start < 0: res = abs(start) 
            else: res = 0
            start = max(0, start)

            end = i+res+(step_size*num_nbs)//2
            
            if end >= len(view_list): 
                res = end-len(view_list)
                start -= res
            end = min(len(view_list), end) 
            if(len(view_list[start:end:step_size]) != (num_nbs)):
                print(i, len(view_list[start:end:step_size]), start, end, len(view_list), view_list[start:end:step_size])
            
            pair_file.write(geomvsnet_utils.get_pair_entry(Path(view_list[i].image_name).stem, num_nbs, [Path(v.image_name).stem for v in view_list[start:end:step_size]], -1000.0))
    