import numpy as np
from . import math_utils

cv_to_nerf_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
cv_to_nerf_mat = np.eye(4,4)
def nerf_from_C2W(C2W):   
	return np.matmul(C2W, cv_to_nerf_mat)


def nerf_from_C2Ws(c2ws:dict):
    nerf_c2ws = {k: nerf_from_C2W(v) for k, v in c2ws.items() }
    return nerf_c2ws

def C2W_from_nerf(nerf_C2W):
    nerf_to_cv_mat = np.linalg.inv(cv_to_nerf_mat)
    # nerf_W2C = np.linalg.inv(nerf_C2W)
    # nerf_W2C =  np.matmul(nerf_W2C, nerf_to_cv_mat)
    # nerf_C2W = np.linalg.inv(nerf_W2C)
    # return nerf_C2W
    return np.matmul(nerf_C2W, nerf_to_cv_mat)

# adapted form instant-ngp:scripts/colmap2nerf.py
def rescaled_nerf_from_C2Ws(c2ws:dict):
    nerf_c2ws = {}
    mean_up = np.zeros(3)
    for k, c2w in c2ws.items():
        c2w[0:3,2] *= -1 # flip the y and z axis
        c2w[0:3,1] *= -1
        c2w = c2w[[1,0,2,3],:]
        c2w[2,:] *= -1 # flip whole world upside down

        mean_up += c2w[0:3,1]
        
        nerf_c2ws[k] = c2w
        
    # rotate up vector to [0,0,1]
    mean_up = mean_up / np.linalg.norm(mean_up)
    R = math_utils.rotmat_from_vec3_to_vec3(mean_up,[0,0,1])
    R = np.pad(R,[0,1])
    R[-1, -1] = 1
    
    # rotate up to be the z axis
    nerf_c2ws = {k: np.matmul(R, c2w) for k, c2w in nerf_c2ws.items()}
   
    # find a central point they are all looking at
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for c2w in nerf_c2ws.values():
        mf = c2w[0:3,:]
        for g in nerf_c2ws.values():
            mg = g[0:3,:]
            p, w = math_utils.closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                totp += p*w
                totw += w
    if totw > 0.0:
        totp /= totw
    # center around totp
    for k, c2w in nerf_c2ws.items():
        nerf_c2ws[k][0:3,3] -= totp

   
    return nerf_c2ws