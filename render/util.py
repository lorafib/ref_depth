# Based on https://github.com/NVlabs/nvdiffrec/blob/slang/render/util.py
# Edits (Copyright Laura Fink, 2024) are tagged with LF


# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import imageio
import cv2

#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

#----------------------------------------------------------------------------
# sRGB color transforms
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def reinhard(f: torch.Tensor) -> torch.Tensor:
    return f/(1+f)

#-----------------------------------------------------------------------------------
# Metrics (taken from jaxNerf source code, in order to replicate their measurements)
#
# https://github.com/google-research/google-research/blob/301451a62102b046bbeebff49a760ebeec9707b8/jaxnerf/nerf/utils.py#L266
#
#-----------------------------------------------------------------------------------

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / np.log(10.) * np.log(mse)

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

#----------------------------------------------------------------------------
# Displacement texture lookup
#----------------------------------------------------------------------------

def get_miplevels(texture: np.ndarray) -> float:
    minDim = min(texture.shape[0], texture.shape[1])
    return np.floor(np.log2(minDim))


# LF: added range option, removed unsqueeze
def tex_2d(tex_map : torch.Tensor, coords : torch.Tensor, filter='nearest', range="11") -> torch.Tensor:
    tex_map = tex_map[None, ...]    # Add batch dimension
    tex_map = tex_map.permute(0, 3, 1, 2) # NHWC -> NCHW
    coords =coords[None, None, ...] 
    if range == "01":
        coords = coords * 2 - 1
    tex = torch.nn.functional.grid_sample(tex_map, coords, mode=filter, align_corners=False)
    tex = tex.permute(0, 2, 3, 1) # NCHW -> NHWC
    return tex[0, 0, ...]

#----------------------------------------------------------------------------
# Cubemap utility functions
#----------------------------------------------------------------------------

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

#----------------------------------------------------------------------------
# Image scaling
#----------------------------------------------------------------------------

def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

# LF: replaced assert with warning on magnification and minification 
def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    if not( (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] <= size[0] and x.shape[2] <= size[1])):
        print("WARNING: Trying to magnify image in one dimension and minify in the other")
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Behaves similar to tf.segment_sum
#----------------------------------------------------------------------------

def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    num_segments = torch.unique_consecutive(segment_ids).shape[0]

    # Repeats ids until same dimension as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:], dtype=torch.int64, device='cuda')).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    result = torch.zeros(*shape, dtype=torch.float32, device='cuda')
    result = result.scatter_add(0, segment_ids, data)
    return result

#----------------------------------------------------------------------------
# Matrix helpers.
#----------------------------------------------------------------------------

def fovx_to_fovy(fovx, aspect):
    return np.arctan(np.tan(fovx / 2) / aspect) * 2.0

def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective_offcenter(fovy, fraction, rx, ry, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)

    # Full frustum
    R, L = aspect*y, -aspect*y
    T, B = y, -y

    # Create a randomized sub-frustum
    width  = (R-L)*fraction
    height = (T-B)*fraction
    xstart = (R-L)*rx
    ystart = (T-B)*ry

    l = L + xstart
    r = l + width
    b = B + ystart
    t = b + height
    
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    return torch.tensor([[2/(r-l),        0,  (r+l)/(r-l),              0], 
                         [      0, -2/(t-b),  (t+b)/(t-b),              0], 
                         [      0,        0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [      0,        0,           -1,              0]], dtype=torch.float32, device=device)

# LF:
# get GL projection matrix from intrinsics parameters
# from livenvs
def perspective_from_f_c(fx, fy, cx, cy, near, far, w, h, device):
    return torch.tensor([[ 2*fx/w,  0.0,    (w - 2*cx)/w, 0.0],
                         [  0.0,    -2*fy/h, (h - 2*cy)/h, 0.0],
                         [  0.0, 0.0, (-far - near) / (far - near), -2.0*far*near/(far-near)],
                         [  0.0, 0.0, -1.0, 0.0]], dtype=torch.float32, device=device)


# LF: 
def scale_intrinsics(fx, fy, cx, cy, width, height, new_width, new_height):
    img_scale = np.array([new_height, new_width])/np.array([height, width])
    width, height = new_width, new_height
    fx, fy = img_scale[1]*fx, img_scale[0]*fy 
    cx, cy = img_scale[1]*cx, img_scale[0]*cy
    return fx, fy, cx, cy, width, height, img_scale



def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1, 0, 0, 0], 
                         [0, c,-s, 0], 
                         [0, s, c, 0], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def scale(s, device=None):
    return torch.tensor([[ s, 0, 0, 0], 
                         [ 0, s, 0, 0], 
                         [ 0, 0, s, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def lookAt(eye, at, up):
    a = eye - at
    w = a / torch.linalg.norm(a)
    u = torch.cross(up, w)
    u = u / torch.linalg.norm(u)
    v = torch.cross(w, u)
    translate = torch.tensor([[1, 0, 0, -eye[0]], 
                              [0, 1, 0, -eye[1]], 
                              [0, 0, 1, -eye[2]], 
                              [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    rotate = torch.tensor([[u[0], u[1], u[2], 0], 
                           [v[0], v[1], v[2], 0], 
                           [w[0], w[1], w[2], 0], 
                           [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    return rotate @ translate

@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)

@torch.no_grad()
def random_rotation(device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.array([0,0,0]).astype(np.float32)
    return torch.tensor(m, dtype=torch.float32, device=device)

#----------------------------------------------------------------------------
# Compute focal points of a set of lines using least squares. 
# handy for poorly centered datasets
#----------------------------------------------------------------------------

def lines_focal(o, d):
    d = safe_normalize(d)
    I = torch.eye(3, dtype=o.dtype, device=o.device)
    S = torch.sum(d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...], dim=0)
    C = torch.sum((d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...]) @ o[..., None], dim=0).squeeze(1)
    return torch.linalg.pinv(S) @ C

#----------------------------------------------------------------------------
# Cosine sample around a vector N
#----------------------------------------------------------------------------
@torch.no_grad()
def cosine_sample(N, size=None):
    # construct local frame
    N = N/torch.linalg.norm(N)

    dx0 = torch.tensor([0, N[2], -N[1]], dtype=N.dtype, device=N.device)
    dx1 = torch.tensor([-N[2], 0, N[0]], dtype=N.dtype, device=N.device)

    dx = torch.where(dot(dx0, dx0) > dot(dx1, dx1), dx0, dx1)
    #dx = dx0 if np.dot(dx0,dx0) > np.dot(dx1,dx1) else dx1
    dx = dx / torch.linalg.norm(dx)
    dy = torch.cross(N,dx)
    dy = dy / torch.linalg.norm(dy)

    # cosine sampling in local frame
    if size is None:
        phi = 2.0 * np.pi * np.random.uniform()
        s = np.random.uniform()
    else:
        phi = 2.0 * np.pi * torch.rand(*size, 1, dtype=N.dtype, device=N.device)
        s = torch.rand(*size, 1, dtype=N.dtype, device=N.device)
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi)*sintheta
    y = np.sin(phi)*sintheta
    z = costheta

    # local to world
    return dx*x + dy*y + N*z

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x : torch.tensor) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Bilinear downsample log(spp) steps
#----------------------------------------------------------------------------

def bilinear_downsample(x : torch.tensor, spp) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    g = x.shape[-1]
    w = w.expand(g, 1, 4, 4) 
    x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    steps = int(np.log2(spp))
    for _ in range(steps):
        xp = torch.nn.functional.pad(x, (1,1,1,1), mode='replicate')
        x = torch.nn.functional.conv2d(xp, w, padding=0, stride=2, groups=g)
    return x.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Singleton initialize GLFW
#----------------------------------------------------------------------------

_glfw_initialized = False
def init_glfw():
    global _glfw_initialized
    try:
        import glfw
        glfw.ERROR_REPORTING = 'raise'
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        test = glfw.create_window(8, 8, "Test", None, None) # Create a window and see if not initialized yet
    except glfw.GLFWError as e:
        if e.error_code == glfw.NOT_INITIALIZED:
            glfw.init()
            _glfw_initialized = True

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, title=None):
    # Import OpenGL
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image[..., 0:3]) if image.shape[-1] == 4 else np.asarray(image)
    height, width, channels = image.shape

    # Initialize window.
    init_glfw()
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.default_window_hints()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save/load helper.
#----------------------------------------------------------------------------

# LF: diff image helper
import matplotlib
import matplotlib.pyplot as plt
diff_cmap = matplotlib.colormaps.get_cmap('magma')
def diff_image(x,y, range=(0.,1.), expand_to=None, cmap=diff_cmap):
    diff = np.linalg.norm(x - y, axis=2)
    diff_mapped = (diff - range[0])/(range[1] - range[0] + 1e-5)
    if cmap == None:
        if expand_to == None:
            expand_to = x.shape[2]
        diff_col = np.repeat(diff_mapped[..., None], expand_to, axis=2)
        if expand_to == None and x.shape[2] == 4:
            diff_col[...,-1] = 1 
    else:   
        diff_col = cmap(diff_mapped)
        if x.shape[2] == 4:
            diff_col[...,-1] = 1 
        else: diff_col = diff_col[...,:3]
    return diff_col

def tensor_to_numpy(t: torch.Tensor, b=0):
    t = t if len(t.shape) < 4 else t[b] 
    return t.detach().flip(0).cpu().numpy()[::-1]

def save_tensor_as_image(fn, t: torch.Tensor, b=0):
    x = tensor_to_numpy(t, b)
    save_image(fn, x)


# LF: exception handling added
def save_image(fn, x : np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except Exception as e:
        print("WARNING: FAILED to save image %s" % fn)
        print(str(e))
        print("shape is", x.shape)
        print("isinf", np.any(np.isinf(x)))
        print("isnan", np.any(np.isnan(x)))
        

def save_image_raw(fn, x : np.ndarray):
    try:
        imageio.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)
        
# LF:         
def save_depth_image(fn, x : np.array, depth_scale=1000.0):
    # np_depth = np.clip(np_depth, 0, 100) # for now clamp to 100m
    x = depth_scale * x
    # Save depth images (convert to uint16)
    cv2.imwrite(str(fn), x.astype(np.uint16))
          

def load_image_raw(fn) -> np.ndarray:
    # return imageio.imread(fn)
    img = cv2.imread(str(fn), cv2.IMREAD_UNCHANGED)
    
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255

#----------------------------------------------------------------------------

def time_to_text(x):
    if x > 3600:
        return "%.2f h" % (x / 3600)
    elif x > 60:
        return "%.2f m" % (x / 60)
    else:
        return "%.2f s" % x

#----------------------------------------------------------------------------

def checkerboard(res, checker_size) -> np.ndarray:
    tiles_y = (res[0] + (checker_size*2) - 1) // (checker_size*2)
    tiles_x = (res[1] + (checker_size*2) - 1) // (checker_size*2)
    check = np.kron([[1, 0] * tiles_x, [0, 1] * tiles_x] * tiles_y, np.ones((checker_size, checker_size)))*0.33 + 0.33
    check = check[:res[0], :res[1]]
    return np.stack((check, check, check), axis=-1)


def quad():
    pos = torch.tensor([[-1.,-1.,0.],[ 1.,-1.,0.], [ 1., 1.,0.], [-1., 1.,0.]]).cuda()
    pos = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)[None, ...]
    pos_idx = torch.tensor([[0,1,2], [0,2,3]], dtype=torch.int32).cuda()
    
    uv = 0.5*pos[:,:,:2]+0.5
    uv_idx = torch.tensor([[0,1,2], [0,2,3]], dtype=torch.int32).cuda()
    return pos, pos_idx, uv, uv_idx


#----------------------------------------------------------------------------
# image filters
#----------------------------------------------------------------------------

from torch.autograd import Variable
import math

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class GaussBlur:
    def __init__(self, num_channels, window_size=11) -> None:
        self.window_size = window_size
        self.channel = num_channels
        self.window = create_window(window_size, self.channel)

    def blur(self, img):
        if img.is_cuda:
            self.window = self.window.cuda(img.get_device())
        self.window = self.window.type_as(img)

        img = torch.nn.functional.conv2d(img, self.window, padding=self.window_size // 2, groups=self.channel)
        return img



def median_filter(input_tensor, kernel_size, handle_zeros=False):
    # Assumes input_tensor is of shape [H, W, C]
    pad = kernel_size // 2
    output = torch.zeros_like(input_tensor)
    
    for c in range(input_tensor.shape[2]):
        # Extract a single channel from the input tensor
        single_channel = input_tensor[:, :, c].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        
        # Pad the single channel
        padded = torch.nn.functional.pad(single_channel, (pad, pad, pad, pad), mode='reflect')
        
        # Unfold the padded tensor to get sliding windows
        unfolded = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        
        # Size of unfolded tensor will be [1, 1, H, W, kernel_size, kernel_size]
        # --> linearize
        linearized_kernel = unfolded.contiguous().view(1, 1, input_tensor.shape[0], input_tensor.shape[1], -1)
        
        # Median 
        linearized_kernel_median = linearized_kernel.median(dim=-1).values
        
        if handle_zeros:
            # fill 0 with furtherst maximum values within kernel
            linearized_kernel_max = linearized_kernel.max(dim=-1).values
            linearized_kernel_median[linearized_kernel_median < 0.0001] = linearized_kernel_max[linearized_kernel_median < 0.0001]
        
        
        # Store the result in the corresponding channel
        output[:, :, c] = linearized_kernel_median.squeeze(0).squeeze(0)

    return output


def create_sobel_kernel(size, axis, normalize=True):
    # Check if size is odd
    if size % 2 == 0:
        raise ValueError("Size must be odd")

    # Create a range tensor from -(size//2) to (size//2)
    range_tensor = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)

    # Create a grid for both dimensions
    if axis == 'x':
        kernel = torch.ger(torch.ones(size), range_tensor)
    elif axis == 'y':
        kernel = torch.ger(range_tensor, torch.ones(size))
        
    # Normalize to have zero sum to maintain brightness
    if normalize:
        sum_kernel = torch.sum(kernel)
        if sum_kernel != 0:
            # subtract mean
            kernel -= sum_kernel / kernel.numel()
    return kernel


class Sobel:
    def __init__(self, window_size=3) -> None:
        self.window_size = window_size
        # Create kernels for x and y directions
        self.kernel_x = create_sobel_kernel(size=window_size, axis='x')
        self.kernel_y = create_sobel_kernel(size=window_size, axis='y')
        self.window = torch.stack([self.kernel_x, self.kernel_y]).unsqueeze(1).cuda()
        print("sobel window", self.window.shape)
    def sobel(self, img):
        if img.is_cuda:
            self.window = self.window.cuda(img.get_device())
        self.window = self.window.type_as(img)
        
        img = torch.nn.functional.conv2d(img, self.window, padding=self.window_size//2)
        return img


class NormalUsingSobel:
    def __init__(self, window_size=3) -> None:
        self.window_size = window_size
        # Create kernels for x and y directions
        self.kernel_x = create_sobel_kernel(size=window_size, axis='x').unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)
        self.kernel_y = create_sobel_kernel(size=window_size, axis='y').unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)

    def calc_normals(self, img):
        img = img[:,:3]
        if img.is_cuda:
            self.kernel_x = self.kernel_x.cuda(img.get_device())
            self.kernel_y = self.kernel_y.cuda(img.get_device())
        self.kernel_x = self.kernel_x.type_as(img)
        self.kernel_y = self.kernel_y.type_as(img)
        
        dz_dx = torch.nn.functional.conv2d(img, self.kernel_x, padding=self.window_size//2, groups=img.shape[1])
        dz_dy = torch.nn.functional.conv2d(img, self.kernel_y, padding=self.window_size//2, groups=img.shape[1])
        
        
        eps=1e-5
        normal = torch.cross(dz_dx, -dz_dy, dim=1)
        # Compute the magnitude of normals
        norm_magnitude = torch.norm(normal, dim=1, keepdim=True)

        # Handle degenerate cases: if magnitude is near zero, use fallback normal
        fallback_normal = torch.tensor([0.0, 0.0, 1.0], device="cuda").view(1, 3, 1, 1)
        normal = torch.where(norm_magnitude > eps, normal / (norm_magnitude + eps), fallback_normal)

        normal[:,0] = -normal[:,0]
        
        # exit(0)
        # normal = torch.stack((-dz_dx, -dz_dy, torch.ones_like(dz_dx)), dim=2)
        # normal = safe_normalize(normal)
        # print(normal.shape)
        # print(normal[5,5])
        return normal

import slangtorch
from pathlib import Path

this_file_path = str(Path(__file__).parent.resolve())
print(this_file_path)
_blur_module = slangtorch.loadModule(this_file_path+"/slang_utils/blur.slang", verbose=True)
class MaskedAtrousFilterFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.tensor, mask: torch.Tensor, weight: torch.Tensor, atrous_step_size: int) -> torch.Tensor:
        input_ = input_.contiguous()
        output = _blur_module.masked_atrous_fwd(
            input_,
            mask, 
            weight, 
            atrous_step_size
        )
        ctx.save_for_backward(input_, mask, weight, output)
        ctx.atrous_step_size = atrous_step_size

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.tensor):
        [input_, mask, weight, output] = ctx.saved_tensors
        input_grad = _blur_module.masked_atrous_bwd(
            input_,
            mask, 
            weight, 
            ctx.atrous_step_size,
            output_grad,
        )
        return input_grad, None, None, None


class AtrousFilterFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.tensor, atrous_step_size: int):
        global savei
        input_ = input_.contiguous()
        output = _blur_module.atrous_fwd(
            input_,
            atrous_step_size
        )
        ctx.save_for_backward(input_, output)
        ctx.atrous_step_size = atrous_step_size

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.tensor):
        [input_, output] = ctx.saved_tensors
        input_grad = _blur_module.atrous_bwd(
            input_,
            ctx.atrous_step_size,
            output_grad,
        )
        return input_grad, None



#----------------------------------------------------------------------------
# transform helpers
#----------------------------------------------------------------------------

cv_flip_mat = torch.tensor([
            [1., 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

def cam_convention_transform(mtx, use_flip_mat, use_rot_mat = True):
    
    # mtx = torch.linalg.inv(mtx)
    if use_flip_mat:
        mtx = torch.matmul(cv_flip_mat.to(device=mtx.device),mtx)   
    if use_rot_mat:
        mtx = mtx @ rotate_x(np.pi / 2, mtx.device)
  
    # mtx = torch.linalg.inv(mtx)
    return mtx
    

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx   = torch.tensor(mtx, dtype=torch.float32).cuda() if isinstance(mtx, np.ndarray) else mtx
    pos     = torch.tensor(pos, dtype=torch.float32).cuda() if isinstance(pos, np.ndarray) else pos
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolution)
    
    if enable_mip:
        texc, texd = dr.interpolate(uv, rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv, rast_out, uv_idx)
        # texc, _ = dr.interpolate(q_uv, rast_out, q_uv_idx)
        color = dr.texture(tex, texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color
