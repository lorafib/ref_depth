import numpy as np
from pathlib import Path

def load_associations(input_path):
    associations = {}
    with open(input_path, "r") as ass:
        for line in ass:
            values = line.split(' ')
            timestamp = float(values[1])
            time_str = f"{timestamp:.05f}"
            associations[time_str] = values[0]
            
    return associations
    
def load_trajectory(input_path):
    trajectory = []
    with open(input_path, "r") as traj:
        for line in traj:
            # timestamp tx ty tz qx qy qz qw
            values = line.split(' ')
            timestamp = float(values[0])
            
            q = np.array([float(values[7]), float(values[4]),float(values[5]), float(values[6])])
            
            trajectory.append({ "T": np.array([float(values[1]),float(values[2]), float(values[3])]),
                                "q": q, 
                                "timestamp": timestamp})
            
    return trajectory

def load_K(input_path):
    if input_path.exists():
        K = np.loadtxt(input_path)
    else:
        print(f"error: {input_path} does not exist!")
    return K

def load_resolution(input_path):    
    if input_path.exists():
        res = np.loadtxt(input_path)
        width, height = int(res[0]), int(res[1])
        return width, height
    else:
        print(f"error: {input_path} does not exist!")
