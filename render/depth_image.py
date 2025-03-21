# Copyright 2024, Laura Fink
 
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt

from . import util
# ln -s /path/to/schnibbets/cv_dataset_utils/ cv_dataset_utils


def estimate_offset_and_scale_from_abs_dm(relative_depth, absolute_depth):
    """
    Estimates offset and scale to map a [0,1]-normalized linear depth map to absolute units 
    
    Parameters:
    - relative_depth: depth map in [0,1] of shape (*, 1).
    - absolute_depth: depth map in R+ of shape (*, 1).
    
    Returns:
    - offset and scale
    """
    
    # strategy 1: rel range in [0,1] offset from min, scale from median
    offset = absolute_depth[absolute_depth.nonzero()].min()
    rel_median = np.median(relative_depth) 
    abs_median = np.median(absolute_depth[absolute_depth.nonzero()]-offset)
    scale = abs_median/rel_median
    
    # # strategy 2: rel range in [0,1]? offset from mean, scale from var
    abs_var, abs_mean = absolute_depth.var(), absolute_depth.mean()
    #     rel_var, rel_mean = relative_depth.var(), relative_depth.mean()
    #     scale = abs_var 
    #     # scale = 0.5*(scale+abs_mean/rel_mean)
    #     offset = abs_mean - scale*rel_mean 
    
    # # strategy 3: rel range in [0,1], offset from min, scale from mean ratio 
    #     offset = absolute_depth[absolute_depth.nonzero()].min()
    #     scale = absolute_depth.mean()/relative_depth.mean()
    
    # # strategy 4: rel range in [0,1], offset from min, scale from max
    #     offset = absolute_depth[absolute_depth.nonzero()].min()
    #     absolute_depth = absolute_depth - offset
    #     scale = absolute_depth.max()

    return offset, scale


def estimate_offset_and_scale_from_pcd(relative_depth, absolute_depth, percentile, align_for, cam_near_far):
    """
    Estimates offset and scale to map a [0,1]-normalized linear depth map to absolute units 
    
    This will align the percentile value and the median of 
    the relative pcd (sampled from the mono dm) to the abs pcd.
    
    Parameters:
    - relative_depth: depth map in [0,1] of shape (N, 1).
    - absolute_depth: depth map in R+ of shape (N, 1).
    
    Returns:
    - offset and scale + some other intermediate results
    """
    
    # if we input a dm for reference, we sitll need to filter out zeros
    # won't affect pcd
    absolute_depth = absolute_depth[absolute_depth.nonzero()]
    
    # we don't want to align anything outside near or far
    # (this should influence only very few values)
    absolute_depth = np.clip(absolute_depth, cam_near_far[0], cam_near_far[1])
    sorted_abs_d = np.sort(absolute_depth.flatten())
    sorted_rel_d = np.sort(relative_depth.flatten())
    
    # sort values to get lowest and highest percentile
    abs_maxo   = sorted_abs_d[int(len(sorted_abs_d) * (1.-percentile))]
    abs_offset = sorted_abs_d[int(len(sorted_abs_d) * (percentile)   )]
    rel_maxo   = sorted_rel_d[int(len(sorted_rel_d) * (1.-percentile))]
    rel_offset = sorted_rel_d[int(len(sorted_rel_d) * (percentile)   )]
    
    rel_median = np.median(relative_depth) 
    abs_median = np.median(absolute_depth)
    
    if align_for == "median":
        scale = (abs_median-abs_offset)/(rel_median-rel_offset)
        offset = abs_median - scale*rel_median

    elif align_for == "max":
        scale = (abs_maxo-abs_offset)/(rel_maxo-rel_offset)
        offset = abs_maxo - scale*rel_maxo
        
    
    # some values for evaluation 
    rel_to_abs = relative_depth*scale + offset
    sorted_rel_to_abs = np.sort(rel_to_abs.flatten())
    rel_to_abs_offset = sorted_rel_to_abs[int(len(sorted_rel_to_abs) * (percentile))]
    rel_to_abs_maxo   = sorted_rel_to_abs[int(len(sorted_rel_to_abs) * (1.-percentile))]
    rel_to_abs_median = np.median(rel_to_abs)
    
    return offset, scale, (abs_offset, abs_median, abs_maxo), (rel_to_abs_offset, rel_to_abs_median, rel_to_abs_maxo)


def estimate_offset_and_scale_from_pcd_lsqr(relative_depth, absolute_depth, cam_near_far):
    """
    Estimates offset and scale to map a [0,1]-normalized linear depth map to absolute units 
    Uses lstsqr to minimize error. 
    NOTE: minimizing error in linear depth space does not yield good results!
    
    Parameters:
    - relative_depth: depth map in [0,1] of shape (N, 1).
    - absolute_depth: depth map in R+ of shape (N, 1).
    
    Returns:
    - offset and scale
    """
    
    absolute_depth = absolute_depth[absolute_depth.nonzero()]
    absolute_depth = np.clip(absolute_depth, cam_near_far[0], cam_near_far[1])
    
    A_mono = torch.tensor(relative_depth, device="cuda", dtype=torch.float32)
    # add homogenous coordinate to allow for offset
    A_mono = torch.cat([A_mono, torch.ones_like(A_mono)], dim=1)
    # print(A_mono.shape)
    B_pcd = torch.tensor(absolute_depth, device="cuda", dtype=torch.float32)
    # B_pcd.unsqueeze_(1)
    # print(B_pcd.shape)

    lsqr_result = torch.linalg.lstsq(A_mono, B_pcd, rcond=2)
    print(lsqr_result)
    
    scale, offset = lsqr_result.solution[0].item(), lsqr_result.solution[1].item()
    
    rel_median = np.median(relative_depth) 
    abs_median = np.median(absolute_depth)
    
    print("abs median:", abs_median)
    print("rel median:", rel_median)
    
    return offset, scale



class DepthImage():
    def __init__(self, args):
        self.args = args
        # actual data
        self.offset_scale = None
        self.relative_depth = None 
        self.valid_mask = None
        
    def from_absolute_depth_offset_scale(self, dm, offset_scale, depth_scale=0.001):
        if type(dm) == str or type(dm) == pathlib.PosixPath or type(dm) == pathlib.WindowsPath :
            self.relative_depth = util.load_image_raw(dm).astype(np.float64)
            self.relative_depth = depth_scale * self.relative_depth
            # to device
            self.relative_depth = torch.tensor(self.relative_depth, dtype=torch.float32).cuda()
        elif type(dm) == np.array:
            # to device
            self.relative_depth = torch.tensor(dm, dtype=torch.float32).cuda()
        elif type(dm) == torch.Tensor:
            self.relative_depth = dm
        else:
            print(f"error: invalid dm type given for from_ablsolute_depth, is {type(dm)}")
        self.relative_depth =  util.scale_img_hwc(self.relative_depth.unsqueeze(-1), self.args.cam_resolution, min="nearest-exact", mag="nearest-exact").squeeze(-1)
        self.valid_mask     = torch.zeros_like(self.relative_depth)
        self.valid_mask[self.relative_depth.nonzero(as_tuple=True)] = 1.0
        
        # normalize between [0,1]
        offset, scale = offset_scale[0], offset_scale[1]
        self.relative_depth = self.relative_depth - offset
        self.relative_depth = self.relative_depth / scale
        self.offset_scale = offset_scale

    
    def from_absolute_depth(self, dm, depth_scale=0.001):
        # absolute_depth = util.load_image_raw((Path(args.input_dir)/depth_path/view.image_name).with_suffix(".png"))
        if type(dm) == str or type(dm) == pathlib.PosixPath or type(dm) == pathlib.WindowsPath :
            self.relative_depth = util.load_image_raw(dm).astype(np.float64)
            self.relative_depth = depth_scale * self.relative_depth
            # to device
            self.relative_depth = torch.tensor(self.relative_depth, dtype=torch.float32).cuda()
        elif type(dm) == np.array:
            # to device
            self.relative_depth = torch.tensor(dm, dtype=torch.float32).cuda()
        elif type(dm) == torch.Tensor:
            self.relative_depth = dm.cuda()
        else:
            print(f"error: invalid dm type given for from_ablsolute_depth, is {type(dm)}")
        self.relative_depth =  util.scale_img_hwc(self.relative_depth.unsqueeze(-1), self.args.cam_resolution, min="nearest-exact", mag="nearest-exact").squeeze(-1)
        self.valid_mask     = torch.zeros_like(self.relative_depth)
        self.valid_mask[self.relative_depth.nonzero(as_tuple=True)] = 1.0
        
        # normalize between [0,1]
        offset = self.relative_depth[self.relative_depth.nonzero(as_tuple=True)].min()
        self.relative_depth = self.relative_depth - offset
        scale = self.relative_depth.max()
        self.relative_depth = self.relative_depth / scale
        self.offset_scale = torch.Tensor([offset, scale])

        
    def from_relative_depth_scale_from_dm(self, rel_dm_path, abs_dm_path, depth_scale=0.001):
        self.relative_depth = util.load_image_raw(rel_dm_path).astype(np.float64)
        # normalize between [0,1]
        self.relative_depth = self.relative_depth - self.relative_depth[self.relative_depth.nonzero()].min()
        self.relative_depth = self.relative_depth / self.relative_depth.max()
        
        # load absolute depth
        absolute_depth = util.load_image_raw(abs_dm_path)
        absolute_depth = depth_scale * absolute_depth

        # estimate scale and offsest from absolute depth
        offset, scale = estimate_offset_and_scale_from_abs_dm(self.relative_depth, absolute_depth)

        # to device        
        self.relative_depth = torch.tensor(self.relative_depth, dtype=torch.float32).cuda()
        self.relative_depth =  util.scale_img_hwc(self.relative_depth.unsqueeze(-1), self.args.cam_resolution, min="nearest-exact", mag="nearest-exact").squeeze(-1)
        self.valid_mask     = torch.ones_like(self.relative_depth, dtype=torch.float32).cuda()
        self.offset_scale   = torch.Tensor([offset, scale]).cuda()
        
        
    def from_relative_depth_scale_from_pcd(self, rel_dm_path, v_points3D, points2D_11, abs_dm_path=None, depth_scale=0.001):
        self.relative_depth = util.load_image_raw(rel_dm_path).astype(np.float64)

        # normalize between [0,1]
        self.relative_depth = self.relative_depth - self.relative_depth[self.relative_depth.nonzero()].min()
        self.relative_depth = self.relative_depth / self.relative_depth.max()
        
        # positive depth values for each 2D point given by the sparse colmap pointcloud
        sparse_ref_depths = -v_points3D[:,:,2, None].squeeze(0).cpu().numpy() 

        # get relative depth values a sparse pcd image positions
        sparse_rel_depth = torch.tensor(self.relative_depth, dtype=torch.float32).cuda()
        sparse_rel_depth = torch.nn.functional.grid_sample(sparse_rel_depth.unsqueeze(0).unsqueeze(0), points2D_11.unsqueeze(0).unsqueeze(0), mode="nearest", align_corners=False)
        sparse_rel_depth = sparse_rel_depth.squeeze(0).squeeze(0).squeeze(0).unsqueeze(-1).cpu().numpy()
        # estimate scale and offsest from absolute depth
        offset, scale, (abs_offset, abs_median, abs_maxo), (rel_to_abs_offset, rel_to_abs_median, rel_to_abs_maxo) = estimate_offset_and_scale_from_pcd(sparse_rel_depth, sparse_ref_depths, self.args.scale_init_percentile, self.args.scale_init_align_for, self.args.cam_near_far)
        # offset, scale = estimate_offset_and_scale_from_pcd_lsqr(sparse_rel_depth, sparse_ref_depths, self.args.cam_near_far)


        ## clamp for near and far
        cl_min = (self.args.cam_near_far[0]+1e-3-offset)/scale
        cl_max = (self.args.cam_near_far[1]-1e-1-offset)/scale
        self.relative_depth = np.clip(self.relative_depth, cl_min, cl_max)
                
        
        # to device        
        self.relative_depth = torch.tensor(self.relative_depth, dtype=torch.float32).cuda()
        self.relative_depth = util.scale_img_hwc(self.relative_depth.unsqueeze(-1), self.args.cam_resolution, min="nearest-exact", mag="nearest-exact").squeeze(-1) 
        self.valid_mask     = torch.ones_like(self.relative_depth, dtype=torch.float32).cuda()
        self.offset_scale   = torch.Tensor([offset, scale]).cuda()
        print("offset, scale:", self.offset_scale)


    def median_filter(self, kernel_size=5):
        self.relative_depth = util.median_filter(self.relative_depth.unsqueeze(-1), kernel_size=kernel_size, handle_zeros=False).squeeze(-1)

        
    def scale(self, scale):
        new_height  = int(self.height*scale)
        new_width   = int(self.width*scale)
        self.relative_depth = util.scale_img_hwc(self.relative_depth[..., None], 
                                size=(new_height, new_width), min="nearest-exact", mag="nearest-exact").squeeze(-1)
        self.valid_mask = util.scale_img_hwc(self.valid_mask[..., None], 
                                size=(new_height, new_width), min="nearest-exact", mag="nearest-exact").squeeze(-1)

                
    @property
    def get_absolute(self, clamp=True):
        relative_depth = self.get_relative
        if clamp: relative_depth = torch.clamp_min(relative_depth, 0)
        absolute_depth = relative_depth * self.offset_scale[1] + self.offset_scale[0]
             
        absolute_depth = absolute_depth*self.valid_mask
        return absolute_depth
    @property
    def get_relative(self):
        return self.relative_depth
        
    @property
    def width(self):
        return self.get_relative.shape[1]
    @property
    def height(self):
        return self.get_relative.shape[0]
    @property
    def absmin(self):
        return self.get_relative.min() * self.offset_scale[1] + self.offset_scale[0]
    @property
    def absmax(self):
        return self.get_relative.max() * self.offset_scale[1] + self.offset_scale[0]
    
    @property
    def relmin(self):
        return self.get_relative.min()
    @property
    def relmax(self):
        return self.get_relative.max()
        
    def plot_histogram(self, title, filename = None):
        return plot_histogram({title: self.get_absolute}, filename)
    
def plot_histogram(data:dict,  filename = None, range = (-0.5, 3.0), bins=None):

    if bins == None:
        bins = int((range[1]-(range[0]))/0.1)
    from matplotlib import rc
    # rc('font',**{'family':'serif','serif':['Times'],'size'   : 14})
    # rc('text', usetex=True)
    cmap = plt.cm.get_cmap('Spectral')
    
    # set font
    # import matplotlib.font_manager
    # print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='otf'))
    # csfont = {'fontname': "Noto Serif"}
    csfont = {}
    fig = plt.figure()
    
    
    for i, (title, values) in enumerate(data.items()):
        if type(values) == torch.Tensor:
            values = values.clone().detach().cpu().numpy().flatten()
        else:
            values = values.flatten()
        
        # if "reference" in data:
        #     valids = data["reference"].nonzero()
        # else:
        #     valids = values.nonzero()
        # values = values[valids]
            
        base_col = cmap(0.25*i)
        hcol = (base_col[0], base_col[1], base_col[2], 0.8*base_col[3] )
        x, bins, p = plt.hist(values, density=True, label=title, bins=bins, range=range, color=hcol)  # arguments are passed to np.histogram
        for item in p:
            item.set_height(item.get_height()/sum(x))
        
        ### plot nth-percentile and median lines
        # get values
        sorted = np.sort(values[values.nonzero()])
        mini =  sorted[int(len(sorted) * (0.001))]
        median =  sorted[int(len(sorted) * (0.5))]
        # set color
        line_c = base_col
        line_c = (0.75*line_c[0], 0.75*line_c[1], 0.75*line_c[2], line_c[3] )
        # plot
        plt.axvline(x = mini,linestyle=":", linewidth=2.,  color =line_c, label = f'0.1th percentile = {mini:.2f}')
        plt.axvline(x = median, linestyle="--", linewidth=2., color = line_c, label = f'median = {median:.2f}')
    
    
    plt.legend(**csfont)
    plt.xlabel("Depth in m", **csfont)
    plt.ylabel("\# Samples", **csfont)
    plt.xlim(range)
    plt.ylim(0., 0.1)
    # plt.ylim(0,self.width*self.height*0.02)
    # plt.title(title,**csfont)
    if filename:
        plt.savefig(filename)
    else:
        plt.show(block=False)
    
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
    plt.close()
    return data