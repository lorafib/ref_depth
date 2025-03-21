# Copyright 2024, Laura Fink

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 

import render.util as util

#############################################################################
# color losses

class VGGPerceptualLoss(nn.Module):
    def __init__(self, inp_scale="-11"):
        super().__init__()
        self.inp_scale = inp_scale
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.vgg = torchvision.models.vgg19(pretrained=True).features

    def forward(self, es, ta):
        self.vgg = self.vgg.to(es.device)
        self.mean = self.mean.to(es.device)
        self.std = self.std.to(es.device)

        if self.inp_scale == "-11":
            es = (es + 1) / 2
            ta = (ta + 1) / 2
        elif self.inp_scale != "01":
            raise Exception("invalid input scale")
        es = (es - self.mean) / self.std
        ta = (ta - self.mean) / self.std

        loss = [torch.abs(es - ta).mean()]
        for midx, mod in enumerate(self.vgg):
            es = mod(es)
            with torch.no_grad():
                ta = mod(ta)

            if midx == 3:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 8:
                lam = 0.75
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 13:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 22:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 31:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
                break
        return loss



vgg_loss = None
def vgg_loss_wrapper_nhwc(pred, tgt):
    if vgg_loss == None:
        vgg_loss = VGGPerceptualLoss(inp_scale="01")
    if len(pred.shape) == 3:    pred = pred[None, ...] 
    if len(tgt.shape) == 3:     tgt = tgt[None, ...] 
    pred =  pred.permute(0,3,1,2)[:,:3,:]
    tgt =   tgt.permute(0,3,1,2)[:,:3,:]
    
    return sum(vgg_loss(pred, tgt))


#############################################################################
# regularizer

def calc_grads(source, target, mask):
    # compute gradients for the blended image along both dimensions for each color channel
    grad_blended_x = torch.diff(target, dim=1, append=target[:, -1, None])
    grad_blended_y = torch.diff(target, dim=0, append=target[-1,None, :])
    
    # compute gradients for the source image along both dimensions for each color channel
    grad_source_x = torch.diff(source * mask, dim=1, append=(source * mask)[:, -1,None])
    grad_source_y = torch.diff(source * mask, dim=0, append=(source * mask)[-1,None, :])    # Compute gradients for the mask (border of invalid pixels)
    # compute gradients for the mask (border of invalid pixels)
    grad_mask_x = torch.diff(mask, dim=1)
    grad_mask_x = torch.cat([grad_mask_x, grad_mask_x[:, -1, None]], dim=1)
    grad_mask_y = torch.diff(mask, dim=0)
    grad_mask_y = torch.cat([grad_mask_y, grad_mask_y[-1,None, :]], dim=0)
    
    return grad_blended_x, grad_blended_y, grad_source_x, grad_source_y, grad_mask_x, grad_mask_y


class WeightedPoissonBlendLoss(nn.Module):
    def __init__(self, source, grad_blur_window=5, weight_scale = 5):
        super(WeightedPoissonBlendLoss, self).__init__()
        self.weight_scale = weight_scale
        self.source = source
        self.blur_fun = None
        if grad_blur_window > 0:
            self.blur_fun = util.MaskedAtrousFilterFunc.apply
            # self.blur_fun = util.GaussBlur(num_channels=1, window_size=grad_blur_window).blur
        
    def compute_weights(self, source, grad_source, mask):
        # 1: Compute weighting terms, inversely proportional to the magnitude of source image gradients
        # weight = 1.0 / (torch.abs(grad_source) + 1e-8)
        # 2: Compute weighting terms, using exp. does not take non-linear behavior of gradients into account.
        # (neighboring pixels with high z hav high gradient)
        # weight = torch.exp(-self.weight_scale*torch.abs(grad_source))
        # 3: additionally, divide by z + epsilon 
        weight = mask*torch.exp(-self.weight_scale*torch.abs(grad_source/(source*mask+0.00001)))
        # 4: no weight
        # weight = torch.ones_like(mask)
        return weight
  
    def blur(self, img, mask, weight):
        for i in range(1):
            img = self.blur_fun(img, mask, weight, 2**i)
            # img = self.blur_fun(img, 2**i)
        return img


    def forward(self, target, mask):
        
        target = util.scale_img_hwc(target.unsqueeze(-1),self.source.shape[0:2], min="nearest-exact", mag="nearest-exact").squeeze(-1)
        source_mask = torch.zeros_like(mask)
        source_mask[self.source.nonzero(as_tuple=True)] = 1.
        mask = source_mask*mask

        grad_blended_x, grad_blended_y, grad_source_x, grad_source_y, grad_mask_x, grad_mask_y = calc_grads(self.source, target, mask)
        # print(torch.max(grad_blended_x), torch.max(grad_blended_y), torch.max(grad_source_x), torch.max(grad_source_y))
        # print(torch.min(grad_blended_x), torch.min(grad_blended_y), torch.min(grad_source_x), torch.min(grad_source_y))       # set border pixels to zero, will have invalid grads
        grad_blended_x[grad_mask_x.nonzero(as_tuple=True)]  = 0. 
        grad_blended_y[grad_mask_y.nonzero(as_tuple=True)]  = 0. 
        grad_blended_x[grad_mask_y.nonzero(as_tuple=True)]  = 0. 
        grad_blended_y[grad_mask_x.nonzero(as_tuple=True)]  = 0. 
        
        
        # Compute weighting terms
        weight_x = self.compute_weights(self.source, grad_source_x, mask)
        weight_y = self.compute_weights(self.source, grad_source_y, mask)

                                   
        if self.blur_fun != None:
            
            # blurred_mask            = self.blur(mask)
            grad_blended_x   = self.blur(grad_blended_x, mask, weight_x)
            grad_blended_y   = self.blur(grad_blended_y, mask, weight_y)

            grad_source_x   = self.blur(grad_source_x, source_mask, weight_x)
            grad_source_y   = self.blur(grad_source_y, source_mask, weight_y)

        
        # Compute the loss as the sum of differences between the gradients within the mask for each dimension
        loss_x = F.mse_loss(weight_x*grad_blended_x, weight_x*grad_source_x)
        loss_y = F.mse_loss(weight_y*grad_blended_y, weight_y*grad_source_y)
        loss = loss_x + loss_y
        
        # print("weigth shapse:", weight_x.shape, weight_y.shape)
        weight = torch.cat([weight_x,weight_y, torch.zeros_like(weight_y)], dim=-1)
        return loss, weight, grad_blended_x, grad_blended_y, grad_source_x, grad_source_y


def uv_loss(uv):
    uv_x = uv[...,0]*uv.shape[1]
    uv_y = uv[...,1]*uv.shape[0]
    grad_uv_x = torch.diff(uv_x, dim=1, append=uv_x[:, -1, None])
    grad_uv_y = torch.diff(uv_y, dim=0, append=uv_y[-1,None, :])
    # print(grad_uv_x.shape, grad_uv_x)
    # print(grad_uv_y.shape, grad_uv_y)
    # print(grad_uv_x.max(), grad_uv_y.max())
    # print(grad_uv_x.min(), grad_uv_y.min())
    # print(grad_uv_x.mean(), grad_uv_y.mean())
    # if correct, gradient of uvs is always positive
    # negate to get wrong grads being positive 
    grad_uv_x = torch.clamp((grad_uv_x-1.0)**2, 0.0, 3.0)
    grad_uv_y = torch.clamp((grad_uv_y-1.0)**2, 0.0, 3.0)
    
    loss =  0.5*torch.mean(grad_uv_x) + 0.5*torch.mean(grad_uv_y)
        
    return loss

      
blur = util.GaussBlur(1,5)
            
def uv_grad_loss(uv, mask, cam_resolution, scale=3.):
    # tboard.add_image("rast/uv", util.tensor_to_numpy(uv, b), it, dataformats="HWC")
    # util.save_tensor_as_image(f'{args.outdir}/uv.png', uv, b)
    uv = uv.clone()
    uv[...,0]  = uv[...,0]*cam_resolution[1]
    uv[...,1]  = uv[...,1]*cam_resolution[0]
    grad_uv_x = torch.norm(torch.diff(uv, dim=2, append=uv[:, :,-1, None,:2]),dim=-1,keepdim=True)
    grad_uv_y = torch.norm(torch.diff(uv, dim=1, append=uv[:, -1, None,:,:2]),dim=-1,keepdim=True)
    # grad_uv_x = torch.diff(uv, dim=2, append=uv[:, :,-1, None,:2]).min(dim=-1,keepdim=True)[0]
    # grad_uv_y = torch.diff(uv, dim=1, append=uv[:, -1, None,:,:2]).min(dim=-1,keepdim=True)[0]
    grad_uv = torch.minimum(grad_uv_x,grad_uv_y)
    # grad_uv1 = torch.cat([grad_uv, torch.zeros_like(uv)],dim=-1)[...,:3]
    # util.save_tensor_as_image(f'{args.outdir}/grad_uv_x.png', grad_uv1, b)
    grad_uv = 1. - torch.clamp_max(scale*grad_uv*grad_uv, 1.)
    grad_uv = torch.clamp_min(grad_uv-0.01, 0.0)
    
    grad_uv = blur.blur(grad_uv.permute(0,3,1,2)).permute(0,2,3,1)
    mask = blur.blur(mask.permute(0,3,1,2)).permute(0,2,3,1)
    grad_uv = grad_uv*mask
    return torch.mean(grad_uv), grad_uv

def uv_to_edges_loss(edge_map, uvs):
    edginess = torch.nn.functional.grid_sample(edge_map, uvs.unsqueeze(0).unsqueeze(0), align_corners=False)
    return torch.mean(edginess) # do not use sqr, edginees may be negative

def smooth_loss(dm, mask):
    filtered_dm = blur.blur(dm.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach()
    sm_mask = blur.blur(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)      
    sm_mask[sm_mask < 1.] = 0. # all values affected by invalid values are masked out
    smooth_loss = ((filtered_dm - dm)/torch.clamp_min(dm, 0.01))**2
    smooth_loss =  sm_mask*smooth_loss

    loss = torch.mean(smooth_loss)
    return loss
    
    
    

#############################################################################
# geometry losses
import numpy as np
import matplotlib.pyplot as plt

class FeaturePointLoss(nn.Module):
    def __init__(self, v_points3D, points2D_11, col3D, huber_delta=0.5):
        super(FeaturePointLoss, self).__init__()
        self.points2D_11 = points2D_11 # normalized to -1, 1 with (-1,-1) being the lower left corner of pixel (0,0)
        self.sparse_ref_depths = -v_points3D[:,:,2, None].squeeze(0) # positive depth values for each 2D point fiven by the sparse colmap pointcloud
        self.loss_fun = torch.nn.HuberLoss(delta=huber_delta, reduction="none")
        
        # for visualization only
        self.col3D = col3D
        
    def sample_depths(self, abs_depth):
        depths = torch.nn.functional.grid_sample(abs_depth.unsqueeze(0).unsqueeze(0), self.points2D_11.unsqueeze(0).unsqueeze(0), mode="nearest", align_corners=False)
        depths = depths.squeeze(0).squeeze(0).squeeze(0).unsqueeze(-1)
        return depths
    
    def loss(self, depths):
        loss = torch.mean(self.loss_fun(self.sparse_ref_depths, depths)/self.sparse_ref_depths)
        return loss
        
    def forward(self, abs_depth):
        depths = self.sample_depths(abs_depth)
        l= self.loss(depths)
        return l
    
    @torch.no_grad
    def plot(self, dm_ref_depths, abs_depth, more_depths={}, filename=None):# ):
        depths = self.sample_depths(abs_depth)
        
        ref_points = torch.cat([self.points2D_11, torch.clamp_max(self.sparse_ref_depths, depths.max()+1.)], dim=1).cpu().numpy()
        cur_points = torch.cat([self.points2D_11, depths], dim=1).cpu().numpy()
        ref_from_dm_points = torch.cat([self.points2D_11, dm_ref_depths], dim=1).cpu().numpy()
        
        for name in more_depths:
            more_depths[name] = torch.cat([self.points2D_11, more_depths[name]], dim=1).cpu().numpy()
        
        print("ref dm vs sparse ref dpeth in percentage of ref dm:",(dm_ref_depths-self.sparse_ref_depths)/dm_ref_depths)
        
        # Create a new figure
        fig = plt.figure()

        # Add a 3D subplot
        ax = fig.add_subplot(111, projection='3d')
    
        # Create scatter plot
        # ax.scatter(data[:, 0], data[:, 1], data[:, 2], **color_config)  # data[:, 0] is x, data[:, 1] is y, data[:, 2] is z
        redish = self.col3D.copy()
        redish[:,0] = np.minimum(redish[:,0]*2, 255.0)
        greenish = self.col3D.copy()
        # greenish[:,1] = np.minimum(greenish[:,1]*2, 255.0)
        blue = self.col3D.copy()
        blue[:,2] = np.minimum(blue[:,2]*2, 255.0)
        whiteish = self.col3D.copy()
        whiteish = np.minimum(whiteish*2, 255.0)
        
        ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], label="refernce", facecolors=greenish/255.0)
        # ax.scatter(ref_from_dm_points[:, 0], ref_from_dm_points[:, 1], ref_points[:, 2], label="refernce from dm", facecolors=whiteish/255.0)
        # ax.scatter(cur_points[:, 0], cur_points[:, 1], cur_points[:, 2], label="current", facecolors=redish/255.0)
        # for name in more_depths:
        #     ax.scatter(more_depths[name][:, 0], more_depths[name][:, 1], more_depths[name][:, 2], label=name, facecolors=blue/255.0)
        ax.legend()
        # Optional: Set labels and title
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Z coordinate')
        ax.set_title('3D Scatter Plot')

        # Show plot
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
           
# from DSINE/utils/utils.py
def compute_normal_error(pred_norm, gt_norm):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    # pred_error = torch.acos(pred_error) #* 180.0 / np.pi
    pred_error = -pred_error
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    return pred_error

class NormalSimilarityLoss(nn.Module):
    def __init__(self, args, input_normal_map, input_certainty_map):
        super(NormalSimilarityLoss, self).__init__()
        # print(util.load_image_raw(input_normal_map).dtype)
        self.normal_map         = torch.tensor(np.array(util.load_image_raw(input_normal_map), dtype=np.float64)).to(device="cuda") / 65535.
        self.certainty_map      = torch.tensor(np.array(util.load_image_raw(input_certainty_map), dtype=np.float64)).to(device="cuda").unsqueeze(-1) / 65535
        self.normal_map         = 2*self.normal_map.to(dtype=torch.float32)     - 1
        self.certainty_map      = self.certainty_map.to(dtype=torch.float32) 
        
        self.normal_map         = util.scale_img_hwc(self.normal_map, args.cam_resolution, min="nearest-exact", mag="nearest-exact")
        self.certainty_map      = util.scale_img_hwc(self.certainty_map, args.cam_resolution, min="nearest-exact", mag="nearest-exact")
        self.normal_map         = self.normal_map.unsqueeze(0).permute(0,3,1,2)[:,:,4:-4,4:-4]
        self.certainty_map      = self.certainty_map.unsqueeze(0).permute(0,3,1,2)[:,:,4:-4,4:-4]


        normal_reg = util.tensor_to_numpy(self.normal_map.permute(0,2,3,1).squeeze(0)) 

        normal_reg = 0.5*normal_reg+0.5
        util.save_image(f'{args.outdir}/dm/normals_reg.png', normal_reg)
        
        normal_reg = util.tensor_to_numpy(self.certainty_map.repeat(1,3,1,1).permute(0,2,3,1).squeeze(0)) 
        util.save_image(f'{args.outdir}/dm/normals_certainty.png', normal_reg)

    def forward(self, pred_norm, mask):
        
        if (torch.any(pred_norm.isinf()) or torch.any(pred_norm.isnan())):
            print("invalid values for pred_norm inf / nan: ", torch.any(pred_norm.isinf()), torch.any(pred_norm.isnan()))
            exit()
        if (torch.any(self.normal_map.isinf()) or torch.any(self.normal_map.isnan())):
            print("invalid values for self.normal_map inf / nan: ", torch.any(self.normal_map.isinf()), torch.any(self.normal_map.isnan()))
            exit()

        loss = mask[:,:,4:-4,4:-4]*self.certainty_map*compute_normal_error(pred_norm[:,:,4:-4,4:-4], self.normal_map)
        if (torch.any(loss.isinf()) or torch.any(loss.isnan())):
            print("invalid values for loss inf / nan: ", torch.any(loss.isinf()), torch.any(loss.isnan()))
            exit()

        # util.save_tensor_as_image("n_loss.png", torch.cat([loss.permute(0,2,3,1), -loss.permute(0,2,3,1),(1-mask[:,:,4:-5,4:-5]*self.certainty_map).permute(0,2,3,1)], dim=-1))

        return torch.mean(loss)