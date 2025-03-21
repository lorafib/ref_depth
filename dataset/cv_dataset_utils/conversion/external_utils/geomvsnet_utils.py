
from pathlib import Path

def write_view_file(output_path, extrinsic, intrinsic, depth_min, depth_max, depth_interval, depth_steps):
    with open(output_path, "w") as cam_file:
        cam_file.write("extrinsic\n")
        cam_file.write(str(extrinsic).replace("[", "").replace("]", ""))
        cam_file.write("\n")        
        cam_file.write("\n")
        
        cam_file.write("intrinsic\n")
        cam_file.write(str(intrinsic[:3,:3]).replace("[", "").replace("]", ""))
        cam_file.write("\n")        
        cam_file.write("\n")
        
        cam_file.write(f"{depth_min} {depth_interval} {depth_steps} {depth_max}\n")
        
        
        
def get_pair_entry(index, num_nbs, nbs: list, placeholder_val) -> str:
    pair_str = f"{index}\n"
    pair_str += f"{num_nbs} "
    for i in nbs:
        pair_str += f"{i} {placeholder_val} "
    pair_str += "\n"
    return pair_str