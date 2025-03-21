import numpy as np
import glob
from pathlib import Path

try:
    import external_utils.math_utils as math_utils
except:
    from .external_utils import math_utils

# This is a basic pinhole camera
class BaseCamera():
    def __init__(self, id, fx, fy, cx, cy, width, height) -> None:
        self.id = id
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    def K(self): 
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = self.fx, self.fy, self.cx, self.cy
        return K
    
    def scale_intrinsics(self, new_width, new_height):
        img_scale = np.array([new_height, new_width])/np.array([self.height, self.width])
        self.width, self.height = new_width, new_height
        self.fx, self.fy = img_scale[1]*self.fx, img_scale[0]*self.fy 
        self.cx, self.cy = img_scale[1]*self.cx, img_scale[0]*self.cy
        return img_scale



def camera_from_K(id, K, width, height) -> BaseCamera:
    camera = BaseCamera(id, K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height)
    return camera
        

# This is the pose corresponding to an image
# R and T follow the COLMAP convention (R and T describe the world2camera transform)
class BaseView():
    def __init__(self, id, R, T, camera:BaseCamera, image_name) -> None:
        self.id = id
        self.R = R
        self.T = T
        self.camera = camera
        self.image_name = image_name # stem+extension, without parent dir!

    def W2C(self):
        w2c = np.eye(4,4)
        w2c[:3,:3] = self.R
        w2c[:3, 3] = self.T
        return w2c

    def C2W(self):
        R,T = math_utils.inv_R_T(self.R, self.T)
        c2w = np.eye(4,4)
        c2w[:3,:3] = R
        c2w[:3, 3] = T
        return c2w
    
    def pos(self):
        _, T = math_utils.inv_R_T(self.R, self.T)
        return T
    
    def view_dir(self):
        R, _ = math_utils.inv_R_T(self.R, self.T)
        return R[:,2]

def view_from_W2C(id, W2C, camera, image_name) -> BaseView:
    return BaseView(id=id, R=W2C[:3,:3], T=W2C[:3, 3], camera=camera, image_name=image_name)

def view_from_C2W(id, C2W, camera, image_name) -> BaseView:
    R,T = math_utils.inv_R_T(C2W[:3,:3], C2W[:3,3])
    return BaseView(id=id, R=R, T=T, camera=camera, image_name=image_name)


class BaseScene():
    def __init__(self) -> None:
        self.cameras = {}
        self.views = {}
        self.color_path = Path()
        
    def set_image_extension_from_color_images(self, abs_color_path):
        files = glob.glob('*', root_dir=str(abs_color_path))
        assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (self.color_path)
        ext_found = False
        for file in files:
            file = Path(file)
            if (abs_color_path/file).is_file():
                if file.suffix.lower() == ".png" or file.suffix.lower() == ".jpg" or file.suffix.lower() == ".jpeg":
                    self.replace_image_extension(file.suffix)
                    ext_found = True
                    print("set suffix of views to", file.suffix.lower())
                    break
            else: print( f"{file} not a files")
        if not ext_found: print(f"error: {file.suffix.lower()} does not seem to be a valid image extension.")  
        
    def replace_image_extension(self, file_extension):
        for k in self.views:
            self.views[k].image_name = str(Path(self.views[k].image_name).with_suffix(file_extension)) 
            
    def find_view_from_name(self, view_name:str):
        # get view
        view = None
        candidates = []
        for v in self.views.values():
            if str(v.image_name).lower().endswith(view_name.lower()): 
                candidates.append(v)
        if len(candidates) == 0: print("error:", view_name, "not found!")
        # return shortest 
        minlen = 1000
        for c in candidates:
            if len(c.image_name) < minlen:
                view = c 
                minlen = len(c.image_name)
        return view
    
    

class DepthMapScene(BaseScene):
    def __init__(self) -> None:
        super(DepthMapScene, self).__init__()
        self.depth_cameras = {}
        self.depth_views = {}
        self.depth_path = Path()

