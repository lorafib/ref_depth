import os
import sys
import glob
import shutil

def main(base_path):
    # load associations
    associations = {}
    ass_file = os.path.join(base_path, "associations.txt")
    with open(ass_file, "r") as ass:
        for line in ass:
            values = line.split(' ')
            timestamp = float(values[1])
            time_str = f"{timestamp:.05f}"
            associations[time_str] = values[0]
        
    # load tum trajectory file for poses
    keyframe_img_names = []
    traj_file = os.path.join(base_path, "KeyframeTrajectory.txt")
    with open(traj_file, "r") as traj:
        for line in traj:
            # timestamp tx ty tz qx qy qz qw
            values = line.split(' ')
            timestamp = float(values[0])
            time_str = f"{timestamp:.05f}"
            keyframe_img_names += [associations[time_str]]
            
    # get color img paths
    all_img_paths = {
        os.path.split(x)[1].split(".")[0] : x
        for x in glob.glob(os.path.join(base_path, "color", "*"))
        if (x.endswith(".jpg") or x.endswith(".png"))
    }
    
    keyframe_img_paths = []
    for i in range(len(keyframe_img_names)):
        keyframe_img_paths += [all_img_paths[keyframe_img_names[i]]]
          
    print("num_keyframes:", len(keyframe_img_paths))
    
    # copy files
    
    print("Copying...")
    dest_dir = os.path.join(base_path, "color_keyframes")
    os.makedirs(os.path.join(base_path, "color_keyframes"), exist_ok=True)
    for k in keyframe_img_paths:
        shutil.copy2(k, dest_dir)
    print("Dope")
        
    return    


if __name__ == "__main__":
    main(sys.argv[1])