# Copyright 2024, Laura Fink

import torch
import numpy as np
import imageio

import render.util as util
from render.depth_image import plot_histogram
from dataset.cv_dataset_utils.depth_maps.metrics_for_depth_maps import calc_metrics, dm_color_map, save_dm_color
from render.rgbd_mesh import calc_edge_map

def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

min_metrics = {}
def log_metrics_with_args(tboard, it, to_be_logged, args, current_metrics):
    converted_args = {} 
    for k,v in args.__dict__.items(): 
        if k in to_be_logged:
            if type(v) in [int, float, str, torch.Tensor, bool]:
                converted_args["hparams/"+k] = v
            else:
                converted_args["hparams/"+k] = str(v)
                
    for k, v in current_metrics.items():
        min_k_str = "MIN_"+k
        if min_k_str in min_metrics:
            min_metrics[min_k_str] = min(min_metrics[min_k_str], current_metrics[k])
        else: min_metrics[min_k_str] = current_metrics[k]
        
    all_metrics = {**min_metrics , **current_metrics}
    all_metrics = {"metrics/"+k :v for k, v in all_metrics.items()}
    # print(all_metrics)
            
    tboard.add_hparams(hparam_dict=converted_args, metric_dict=all_metrics,run_name="hparams", global_step=it)
    
calc_normals = util.NormalUsingSobel(5).calc_normals
def write_depthmap_output(args, it, stage, rgbd_mesh_opt, rgbd_mesh_ref, tboard):
    def add_image_to_tboard(label, img):
        if type(img) == torch.Tensor:
            img = util.scale_img_hwc(img, size=(img.shape[0]//2,img.shape[1]//2 ))
        tboard.add_image(label, img, it, dataformats="HWC")
        
    depth_image_opt = rgbd_mesh_opt.depth_image
    d_opt           = depth_image_opt.get_absolute
    if args.ref_dm_available:
        depth_image_ref = rgbd_mesh_ref.depth_image
        d_opt = util.scale_img_hwc(d_opt.unsqueeze(-1),(depth_image_ref.height, depth_image_ref.width)).squeeze(-1)
        d_ref = depth_image_ref.get_absolute

        d_opt_clamped = depth_image_ref.valid_mask*torch.clamp_max(d_opt, 10.0)
        d_ref_clamped = torch.clamp_max(d_ref, 10.0)
        add_image_to_tboard("depth/diff (clamped to 10m)", torch.abs(d_opt_clamped - d_ref_clamped).unsqueeze(-1))
    
        # within threshold map
        tau = 0.05
        abs_depth_diff        = torch.abs(d_opt - d_ref)[..., None]
        within_threshold_map  = torch.ones_like(abs_depth_diff)
        within_threshold_map[abs_depth_diff > tau] = 0
        within_threshold_map = torch.cat([1-within_threshold_map, within_threshold_map, torch.zeros_like(within_threshold_map)], -1)
        within_threshold_map = depth_image_ref.valid_mask[..., None]*within_threshold_map
        add_image_to_tboard(f"depth/accurate (tau={tau})", within_threshold_map)
        
    # plot histogram
    # plot_img = plot_histogram({"Reference":depth_image_ref.get_absolute, "Current":depth_image_opt.get_absolute}, filename=f"{args.outdir}/histogram_depth_current.png")
    # add_image_to_tboard("hist/depth current", plot_img)
    
    # write depth visualizations
    add_image_to_tboard("depth/depth current (white == 20m)", torch.clamp(depth_image_opt.get_absolute.unsqueeze(-1)/20,0,1))
    if args.ref_dm_available: add_image_to_tboard("depth/depth ref (white == 20m)",     torch.clamp(depth_image_ref.get_absolute.unsqueeze(-1)/20,0,1))
    def ugly_visualization(d):
        vis_depth = (d.unsqueeze(-1)).repeat([1,1,3])
        vis_depth[..., 0] = torch.frac(vis_depth[..., 0])
        vis_depth[..., 1] = torch.frac(vis_depth[..., 1]*10)
        vis_depth[..., 2] = torch.clamp(vis_depth[..., 2]/20, 0,1)
        return vis_depth
    add_image_to_tboard("depth/depth vis (white == 20m)", ugly_visualization(d_opt))
    if args.ref_dm_available: add_image_to_tboard("depth/depth ref vis (white == 20m)", ugly_visualization(d_ref))
    # write actual depthmap
    util.save_depth_image(f'{args.outdir}/dm/depth_abs.png', util.tensor_to_numpy(depth_image_opt.get_absolute))
    
    # render geometry information
    out_mesh =  rgbd_mesh_opt.render_self(out_attr=["depth", "uv", "vpos", "rast_out", "color", "wireframe"])
    
    # current depth from mesh 
    depth_from_mesh =     util.tensor_to_numpy(out_mesh["depth"])
    util.save_depth_image(f'{args.outdir}/dm/depth_{stage}.png', depth_from_mesh)
    save_dm_color(f'{args.outdir}/dm/depth_colored_{it+1:08d}.png', depth_from_mesh, 0.1, 3.0)
    util.save_depth_image(f'{args.outdir}/dm/depth.png', depth_from_mesh)
    
    # compare current rendered mesh against state in depth image (for safety)
    # depth_from_mesh_scaled =     util.tensor_to_numpy(util.scale_img_nhwc(depth_from_mesh, (depth_image_opt.height, depth_image_opt.width)))
    # diff_rendered_vs_abs = util.diff_image(depth_from_mesh_scaled, util.tensor_to_numpy(depth_image_opt.get_absolute)[...,None])
    # util.save_image(f'{args.outdir}/dm/depth_renderedvsimage.png', diff_rendered_vs_abs)
    
    # uv
    uv =    out_mesh["uv"]
    uv =    torch.cat([uv, torch.zeros_like(uv)[...,:1]], dim=-1)
    util.save_image(f'{args.outdir}/dm/uv.png', util.tensor_to_numpy(uv))
    # barycentric coords
    rast_out = out_mesh["rast_out"]
    rast_out[:,:,:,2] = rast_out[:,:,:,3]/rast_out[:,:,:,3].max()
    util.save_tensor_as_image(f'{args.outdir}/dm/tris.png', rast_out)
    add_image_to_tboard("depth/tris", rast_out[0])
    
    # wireframe
    # wire_col = torch.ones_like(out_mesh["color"][...,:3])*torch.tensor([0.62, 0.004, 0.26]).cuda()
    wire_col = torch.ones_like(out_mesh["color"][...,:3])*0.7*torch.tensor([0.33, 0.626, 0.824]).cuda()
    wireframe = out_mesh["color"][...,:3]*out_mesh["wireframe"]+ (1.-out_mesh["wireframe"]) * wire_col
    util.save_tensor_as_image(f'{args.outdir}/dm/wireframe.png', wireframe)
    
    # edgemap
    add_image_to_tboard("depth/cur_edge_map", rgbd_mesh_opt.edge_map.squeeze(0))
    print("edge map res: ", rgbd_mesh_opt.edge_map.shape)
    util.save_tensor_as_image(args.outdir+"/dm/edge_map.png", rgbd_mesh_opt.edge_map.repeat(1,1,1,3))

    
    
    
            
    normal = util.tensor_to_numpy(calc_normals(out_mesh["vpos"].permute(0,3,1,2)).permute(0,2,3,1).squeeze(0)) 
    print(normal.min(), normal.max(), np.mean(normal))
    normal = 0.5*normal+0.5
    print(normal.min(), normal.max(), np.mean(normal))
    util.save_image(f'{args.outdir}/dm/normals.png', normal)
    
    # render ref depth map
    if args.ref_dm_available:  
        out_mesh =  rgbd_mesh_ref.render_self()
        depth =     util.tensor_to_numpy(rgbd_mesh_ref.render_self(["depth"])["depth"])
        util.save_depth_image(f'{args.outdir}//dm/depth_ref.png', depth)
        
        normal = util.tensor_to_numpy(calc_normals(out_mesh["vpos"].permute(0,3,1,2)).permute(0,2,3,1).squeeze(0)) 
        normal = 0.5*normal+0.5
        util.save_image(f'{args.outdir}/dm/normals_ref.png', normal)
    
    

def get_img_progress(args, it, b, out, out_opt, target):
        
    #col 
    img_b =     util.tensor_to_numpy(out["color"], b)[...,:3]
    img_o =     util.tensor_to_numpy(out_opt["color"], b)[...,:3]
    img_tgt =   util.tensor_to_numpy(target["img"], b)[...,:3]
    #pos 
    pos_b =     util.tensor_to_numpy(out["vpos"], b)[...,:3]
    pos_o =     util.tensor_to_numpy(out_opt["vpos"], b)[...,:3]
    # diffs
    diff_col =      util.diff_image(img_b, img_o)
    diff_tgt_col =  util.diff_image(img_tgt*util.tensor_to_numpy(out_opt["mask"], b), img_o)
    diff_pos =      util.diff_image(pos_b[...,2,None], pos_o[...,2,None])
    
    img_stack = [img_o, img_b, img_tgt, diff_col, diff_tgt_col, diff_pos]
    try:
        result_image = make_grid(np.stack(img_stack), ncols=3)
        # if display_image:
        #     util.display_image(result_image, size=args.cam_resolution, title='%d / %d' % (it, args.max_iter))
        # if args.mp4save_interval:
    except  Exception as e:
        print(repr(e)) 
        for i,im in enumerate(img_stack): print(i,":",im.shape)
    return result_image

def write_img_output(args, it, b, out, out_opt, target, tboard):
    def add_image_to_tboard(label, img):
        if type(img) == torch.Tensor:
            img = util.scale_img_hwc(img, size=(img.shape[0]//2,img.shape[1]//2 ))
        tboard.add_image(label, img, it, dataformats="HWC")
        
    img_o =     util.tensor_to_numpy(out_opt["color"], b)[...,:3]
    img_tgt =   util.tensor_to_numpy(target["img"], b)[...,:3]
    pos_o =     util.tensor_to_numpy(out_opt["vpos"], b)[...,:3]
    
    diff_tgt_col =  util.diff_image(img_tgt*util.tensor_to_numpy(out_opt["mask"]*out_opt["edge_map"], b), img_o*util.tensor_to_numpy(out_opt["mask"]*out_opt["edge_map"],b))

    # ref 
    if args.ref_dm_available: 
        img_b =     util.tensor_to_numpy(out["color"], b)[...,:3]
        pos_b =     util.tensor_to_numpy(out["vpos"], b)[...,:3]
        diff_col =      util.diff_image(img_b, img_o)
        diff_pos =      util.diff_image(pos_b[...,2,None], pos_o[...,2,None])
        tboard.add_images("color/batch", util.scale_img_nhwc(torch.cat([out_opt["color"], out["color"],target["img"]], dim=2)[...,:3], (9*50,3*16*50)), it, dataformats="NHWC")
        
    
    # triangle visualization
    rast_out = out_opt["rast_out"]
    rast_out[:,:,:,2] = rast_out[:,:,:,3]/rast_out[:,:,:,3].max()
    util.save_tensor_as_image(f'{args.outdir}/rast_out.png', rast_out, b)
    add_image_to_tboard("rast/current", rast_out[b])
    
    # # edge map visualization 
    # negative part: red , positive part: green channel
    edge_map_vis_col = torch.cat([torch.clamp(-out_opt["edge_map"], 0., 1.), torch.clamp(out_opt["edge_map"], 0., 1.), torch.zeros_like(out_opt["edge_map"])], dim=-1)
    edge_map_vis_col = 0.5*(out_opt["color"][...,:3] + edge_map_vis_col)
    util.save_tensor_as_image(f'{args.outdir}/color_masked_edge_map.png', edge_map_vis_col, b)
    add_image_to_tboard("shadow/color masked by edge_map ", edge_map_vis_col[b])
    
    util.save_tensor_as_image(args.outdir+"/edge_map.png", out_opt["edge_map"].repeat(1,1,1,3))


    # write color stuff
    util.save_image(f'{args.outdir}/color_optim.png', img_o)
    util.save_image(f'{args.outdir}/color_optim_masked.png', np.concatenate([img_o*util.tensor_to_numpy(out_opt["edge_map"], b), util.tensor_to_numpy(out_opt["mask"], b)], axis=-1))
    util.save_image(f'{args.outdir}/color_optim_alphamasked.png', np.concatenate([img_o*util.tensor_to_numpy(out_opt["edge_map"], b), util.tensor_to_numpy(out_opt["mask"]*out_opt["edge_map"], b)], axis=-1))
    
    util.save_image(f'{args.outdir}/color_tgt.png', img_tgt)
    add_image_to_tboard("color/current", img_o)
    add_image_to_tboard("color/target", img_tgt)
    
    # write diff stuff
    add_image_to_tboard("color/diff tgt", diff_tgt_col)
    util.save_image(f'{args.outdir}/diff_tgt_col.png', np.concatenate([diff_tgt_col, util.tensor_to_numpy(out_opt["mask"], b)], axis=-1))

    # ref 
    if args.ref_dm_available: 
        util.save_image(f'{args.outdir}/color_ref.png', img_b)
        add_image_to_tboard("color/ref", img_b)
        util.save_image(f'{args.outdir}/diff_col.png', diff_col)
        add_image_to_tboard("color/diff ref", diff_col)
        util.save_image(f'{args.outdir}/diff_pos.png', diff_pos)
    
    
    # write neural feat stuff
    if args.use_feat_enc:
        util.save_tensor_as_image(f'{args.outdir}/feat03_opt.png', out_opt["feat"][...,:3], b)
        util.save_tensor_as_image(f'{args.outdir}/feat-3_opt.png', out_opt["feat"][...,-3:], b)
        util.save_tensor_as_image(f'{args.outdir}/feat03_tgt.png', target["feat"][...,:3], b)
        util.save_tensor_as_image(f'{args.outdir}/feat-3_tgt.png', target["feat"][...,-3:], b)
        



writers = {}
def write_anim(args, name:str, frame, b=0):
    if not name in writers:
        writers[name] =  imageio.get_writer(f'{args.outdir}/anims/{name}.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')

    
    if type(frame) == torch.Tensor:
        frame =     util.tensor_to_numpy(frame, b)[...,:3]
    else:
        frame =     frame[...,:3]
        
    writers[name].append_data(np.clip(np.rint(frame*255.0), 0, 255).astype(np.uint8))

def write_anim_output(args, it, b, out, out_opt, target):
    result_image = get_img_progress(args, it, b, out, out_opt, target)
    write_anim(args, "progress", result_image)
                        #col 
    img_o =     util.tensor_to_numpy(out_opt["color"], b)[...,:3]
    img_tgt =   util.tensor_to_numpy(target["img"], b)[...,:3]
    pos_b =     util.tensor_to_numpy(out["vpos"], b)[...,:3]
    pos_o =     util.tensor_to_numpy(out_opt["vpos"], b)[...,:3]
    diff_tgt_col =  util.diff_image(img_tgt*util.tensor_to_numpy(out_opt["mask"], b), img_o)
    diff_depth =      util.diff_image(pos_b[...,2,None], pos_o[...,2,None])
    write_anim(args, "projected_color_optim", out_opt["color"], b)
    write_anim(args, "projected_diff_tgt_color", diff_tgt_col, b)
    write_anim(args, "projected_diff_depth", diff_depth, b)
    

def write_anim_depthmap_output(args, it, stage, rgbd_mesh_opt, rgbd_mesh_ref, glctx):
        
    depth_image_opt = rgbd_mesh_opt.depth_image
    depth_image_ref = rgbd_mesh_ref.depth_image
    
    # plot histogram
    plot_img = plot_histogram({"Reference":depth_image_ref.get_absolute, "Current":depth_image_opt.get_absolute}, filename=f"{args.outdir}/histogram_depth_current.png", range = (-0.2, 3.), bins=90)
    write_anim(args, "histogram", plot_img/255.0)
    
    d_opt = util.scale_img_hwc(depth_image_opt.get_absolute.unsqueeze(-1),(depth_image_ref.height, depth_image_ref.width)).squeeze(-1)
    d_ref = depth_image_ref.get_absolute
    
    d_opt_clamped = util.tensor_to_numpy(depth_image_ref.valid_mask*torch.clamp_max(d_opt, 10.0))
    d_ref_clamped = util.tensor_to_numpy(torch.clamp_max(d_ref, 10.0))
    diff_depth =      util.diff_image(d_opt_clamped[...,None], d_ref_clamped[...,None])
    write_anim(args, "diff_depth", diff_depth)    

    
    out_mesh =  rgbd_mesh_opt.render(glctx, rgbd_mesh_opt.mvp, args.cam_resolution, out_attr=["vpos", "uv", "rast_out", "color", "wireframe"], enable_mip=False, max_mip_level=0)
    # depth =     util.tensor_to_numpy(out_mesh["vpos"])
    # depth =     -depth[:,:,2, None]
    
    write_anim(args, "depth_colored",       dm_color_map(util.tensor_to_numpy(d_opt), 0.1, 3.0)) 
    write_anim(args, "depth_ref_colored",   dm_color_map(util.tensor_to_numpy(d_ref), 0.1, 3.0))    

    wire_col = torch.ones_like(out_mesh["color"][...,:3])*0.7*torch.tensor([0.33, 0.626, 0.824]).cuda()
    wireframe = out_mesh["color"][...,:3]*out_mesh["wireframe"]+ (1.-out_mesh["wireframe"]) * wire_col
    write_anim(args, "wireframe", wireframe) 
    
    write_anim(args, "edge_map", calc_edge_map(d_opt.unsqueeze(0).unsqueeze(0)).permute(0,2,3,1).squeeze(0))
    write_anim(args, "edge_map_ref", calc_edge_map(d_ref.unsqueeze(0).unsqueeze(0)).permute(0,2,3,1).squeeze(0))

    