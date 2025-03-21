# Copyright 2024, Laura Fink

import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import tqdm
import time

import nvdiffrast.torch as dr

from dataset.cv_dataset_utils.conversion import nerf as nerf_dset
from dataset.cv_dataset_utils.depth_maps.metrics_for_depth_maps import calc_metrics
from dataset.dataset_nerf import DatasetNERF
from dataset.dataset_synthetic import DatasetSynthetic, test_render
from dataset.pointcloud import init_sparse_pcd_from_ref_dm, init_sparse_pcd, calc_pcd_vs_refdm_diff, plot_pcd_diffs_on_img
import render.util as util
from render.rgbd_mesh import RGBD_Mesh, DepthImage, Adaptive_RGBD_Mesh, calc_edge_map
from render.tonemapper import InvertableToneMapper as Tonemapper
from render import loss as Loss
from models import ibrnet_encoder
from output_writing import * 


#----------------------------------------------------------------------------
# Helpers.

def depth_image_name_from_color(input_dir, color_path, depth_dir, view_name):
    depth_path = str(color_path).replace("color", depth_dir)
    depth_path = Path(input_dir)/depth_path/view_name
    depth_path = Path(str(depth_path).replace("jpg", "png")) # depth maps are always pngs
    depth_path = Path(str(depth_path).replace("JPG", "png")) # depth maps are always pngs
    return depth_path

def init_depth_image(args, scene, view, depth_mode, depth_dir, v_points3D=None, points2D_11=None, rendered_depth=None, offset_scale=None):
    depth_image = DepthImage(args=args)
    
    depth_path          = depth_image_name_from_color(args.input_dir, scene.color_path, depth_dir, view.image_name)
    depth_path_of_abs   = depth_image_name_from_color(args.input_dir, scene.color_path, "depth", view.image_name)
    
    if depth_mode == "absolute":
        depth_image.from_absolute_depth(depth_path, args.depth_image_depth_scale)
    elif depth_mode == "relative_to_absdm":
        depth_image.from_relative_depth_scale_from_dm(depth_path, depth_path_of_abs, args.depth_image_depth_scale)
        depth_image.median_filter()
    elif depth_mode == "relative_to_abspcd":
        depth_image.from_relative_depth_scale_from_pcd(depth_path, v_points3D, points2D_11, args.depth_image_depth_scale)
        depth_image.median_filter()
    elif depth_mode == "render":
        depth_image.from_absolute_depth(rendered_depth)
    elif depth_mode == "offset_scale":
        depth_image.from_absolute_depth_offset_scale(depth_path, offset_scale=offset_scale)
    else: print("error:", depth_mode, "is invalid!")
            
    return depth_image


def init_pcd(args, scene, view, depth_image_ref=None):
    v_points3D, points2D_11, col3D = None, None, None
    if args.pcd_available and args.input_colmap_sparse.endswith("from_ref_dm"):
        assert depth_image_ref != None
        print("Load sparse pcd from reference depth.. (args.input_colmap_sparse:", args.input_colmap_sparse, ")")
        v_points3D, points2D_11, col3D = init_sparse_pcd_from_ref_dm(args, view, scene, depth_image_ref, perturb_std=args.sparse_pcd_from_ref_perturb_std, num_points=args.sparse_pcd_from_ref_num_points, outlier_ratio=args.sparse_pcd_from_ref_outlier_ratio)
    elif args.pcd_available:
        print("Load sparse pcd.. (args.input_colmap_sparse:", args.input_colmap_sparse, ")")
        v_points3D, points2D_11, col3D = init_sparse_pcd(args, view, scene)
    else: 
        print("No sparse pcd available. (args.input_colmap_sparse:", args.input_colmap_sparse, ")")
        return v_points3D, points2D_11, col3D
   
    return v_points3D, points2D_11, col3D
    


def refine_depthmap(
             args,        
             log_fn            = None,
             mp4save_fn        = None,
            ):

    # setup output and logging stuff
    log_file = None
    tboard = None
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        os.makedirs(args.outdir+"/dm", exist_ok=True)
        os.makedirs(args.outdir+"/anims/", exist_ok=True)
        
        tboard = SummaryWriter(args.outdir)
        if log_fn:
            log_file = open(f'{args.outdir}/{log_fn}', 'wt')
        
        
    args_table = f"""
    | Argument | Value  |
    |----------|--------|
    """
    for k, v in args.__dict__.items():
        args_table += f"| {k}    | {v}  |\n"
    
    args_table = '\n'.join(l.strip() for l in args_table.splitlines())
    tboard.add_text("Passed arguments", args_table)
    with open(args.outdir+"/config.json", "w") as config_file: 
        json.dump(args.__dict__, config_file)
        
    # Rasterizer context
    print("Init GL Context..")
    glctx = dr.RasterizeGLContext() if args.use_opengl else dr.RasterizeCudaContext()
    print("Done.")

    # get feature encoder
    feat_model = None
    if args.use_feat_enc:
        feat_model = ibrnet_encoder.load_feature_net()
        feat_model.eval()
    
    
    # load camera/ scene parameters
    scene = nerf_dset.load_nerf_as_scene(Path(args.input_dir))
    
    # Create dataset
    init_ref_dm_from = None
    if args.init_depth_mode == "render":
        dataset_train    = DatasetSynthetic(Path(args.input_dir)/ 'transforms.json', args, glctx, examples=(args.max_iter+1), feat_encoder=feat_model)
        dataset_validate = DatasetSynthetic(Path(args.input_dir)/ 'transforms.json', args, glctx, feat_encoder=feat_model)
        # test_render(dataset_validate) # put some test renderings to exisiting dir "synth"
        init_ref_dm_from = "render"
        color_image = dataset_validate[args.dset_ref_index]["img"][0].numpy() # will be provided to the rgbd mesh
        rendered_ref_dm = dataset_validate[args.dset_ref_index]["depth"][0]
    else:
        dataset_train    = DatasetNERF(Path(args.input_dir)/ 'transforms.json', args, examples=(args.max_iter+1), feat_encoder=feat_model)
        dataset_validate = DatasetNERF(Path(args.input_dir)/ 'transforms.json', args, feat_encoder=feat_model)
        init_ref_dm_from = "absolute"
        color_image = None # will be loaded in rgbd_mesh
        rendered_ref_dm = None
    scene.set_image_extension_from_color_images(Path(args.input_dir)/str(scene.color_path))
    view = scene.find_view_from_name(args.view_name)
    print("Provided image name is:", args.view_name, "; Found view is:", view.image_name)
    idx_of_view_in_dataset = dataset_train.get_idx_of_frame(args.view_name)
    print("idx in dataset is:", idx_of_view_in_dataset)
    
    # depth images and meshes 
    print("Init RGBD Meshes (and PCD)...")
    depth_image_ref = None
    rgbd_mesh_ref = None
    if args.ref_dm_available:
        depth_image_ref = init_depth_image(args, scene, view, init_ref_dm_from, args.init_ref_depth_dir, rendered_depth=rendered_ref_dm)
        rgbd_mesh_ref  = RGBD_Mesh(args, glctx, depth_image_ref, scene, view=view, down_scale=1, mode="eval", color_image=color_image) 
    
    # pointcloud init
    # read poin cloud if available, construct pcd from ref dm or return Nones
    v_points3D, points2D_11, col3D = init_pcd(args, scene, view, depth_image_ref)
    
    depth_image_init = init_depth_image(args, scene, view, args.init_depth_mode, args.init_depth_dir, v_points3D, points2D_11, rendered_depth=rendered_ref_dm)   
    depth_image_opt = init_depth_image(args, scene, view, args.init_depth_mode, args.init_depth_dir, v_points3D, points2D_11, rendered_depth=rendered_ref_dm)   
    rgbd_mesh_opt   = Adaptive_RGBD_Mesh(args, glctx, depth_image_opt, scene, view=view, down_scale=args.depth_image_downsample, mode="train_"+args.init_train_mode, color_image=color_image, simplification_ratio=args.init_adaptive_mesh_simplification_ratio) 

    # plot pcd vs reference depth map
    if args.pcd_available and args.ref_dm_available: 
        pcd_diff_to_gt = calc_pcd_vs_refdm_diff(args, points2D_11, v_points3D, depth_image_ref)
        plot_pcd_diffs_on_img(rgbd_mesh_opt.color_image.squeeze(0).cpu().numpy(), points2D_11.cpu().numpy(), pcd_diff_to_gt.cpu().numpy(), filename=f'{args.outdir}/pcd_aligned.png')

    print("Done.")

    
    if args.use_feat_enc:
        rgbd_mesh_opt.c_feat_image, rgbd_mesh_opt.f_feat_image  = feat_model.forward_nhwc(rgbd_mesh_opt.color_image)
        if args.ref_dm_available:
            rgbd_mesh_ref.c_feat_image, rgbd_mesh_ref.f_feat_image = feat_model.forward_nhwc(rgbd_mesh_ref.color_image)
    
    # setup losses
    mse_loss = torch.nn.functional.mse_loss
    
    depth_image_regularizer = init_depth_image(args, scene, view, init_ref_dm_from, args.init_depth_dir, rendered_depth=rendered_ref_dm)
    depth_image_regularizer.median_filter()
    poisson_loss = Loss.WeightedPoissonBlendLoss(depth_image_regularizer.get_relative, args.loss_poisson_blur_window, weight_scale=10)
    
    if args.pcd_available:
        point_regularizer = Loss.FeaturePointLoss(v_points3D, points2D_11, col3D)

    if args.normal_available:
        normal_regularizer = Loss.NormalSimilarityLoss(args, Path(args.input_normal_dir)/view.image_name.replace(".JPG", ".png"), Path(args.input_normal_dir)/view.image_name.replace(".JPG", "_certainty.png"))

    # setup dataloaders
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch, collate_fn=dataset_train.collate, shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)
    
    write_depthmap_output(args, -1, "init", rgbd_mesh_opt, rgbd_mesh_ref, tboard)
    
    
    # setup tonemapper
    if args.use_tonemapper:
        tonemapper = Tonemapper()
        tonemapper.setup_exposure_params(dataset_train.n_images)
        tonemapper = tonemapper.train().cuda()
    
    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    

    def train(start_it, current_stage_index, max_iter, cur_train_mode, lr, cur_loss_color_weight):
        
        print(50*"#")
        print("Train with:", start_it, max_iter, cur_train_mode, lr)
        print(50*"#")

        def setup_optimizer_for_train_mode(): 
            optimizer = None
            # Setup Adam optimizer
            if cur_train_mode == "depth": 
                if type(rgbd_mesh_opt) == Adaptive_RGBD_Mesh: 
                    rgbd_mesh_opt.z_factor.requires_grad = True
                    optimizer    = torch.optim.Adam([rgbd_mesh_opt.z_factor], lr=lr)
                else:   
                    rgbd_mesh_opt.depth_image.relative_depth.requires_grad = True
                    optimizer    = torch.optim.Adam([rgbd_mesh_opt.depth_image.relative_depth], lr=lr)
            elif cur_train_mode == "depth_uv": 
                if type(rgbd_mesh_opt) == Adaptive_RGBD_Mesh: 
                    rgbd_mesh_opt.z_factor.requires_grad = True
                    rgbd_mesh_opt.uv_offset.requires_grad = True
                    optimizer    = torch.optim.Adam([rgbd_mesh_opt.z_factor, rgbd_mesh_opt.uv_offset], lr=lr)
                else: 
                    print("error: invalid train mode for current rgb_mesh type!")
                    exit()
            elif cur_train_mode == "transfer_func_uv":
                if type(rgbd_mesh_opt) == Adaptive_RGBD_Mesh: 
                    rgbd_mesh_opt.uv_offset.requires_grad = True
                    rgbd_mesh_opt.depth_transfer_function.requires_grad = True
                    optimizer    = torch.optim.Adam([rgbd_mesh_opt.uv_offset, *rgbd_mesh_opt.depth_transfer_function.parameters()], lr=lr)
                    # optimizer    = torch.optim.Adam([rgbd_mesh_opt.uv_offset], lr=lr)
                else: 
                    print("error: invalid train mode for current rgb_mesh type!")
                    exit()
            elif cur_train_mode == "transfer_func_depth_uv": 
                if type(rgbd_mesh_opt) == Adaptive_RGBD_Mesh: 
                    rgbd_mesh_opt.z_factor.requires_grad = True
                    rgbd_mesh_opt.uv_offset.requires_grad = True
                    rgbd_mesh_opt.depth_transfer_function.requires_grad = True
                    optimizer    = torch.optim.Adam([rgbd_mesh_opt.z_factor, rgbd_mesh_opt.uv_offset, *rgbd_mesh_opt.depth_transfer_function.parameters()], lr=lr)
                else: 
                    print("error: invalid train mode for current rgb_mesh type!")
                    exit()
            elif cur_train_mode == "transfer_func_depth": 
                if type(rgbd_mesh_opt) == Adaptive_RGBD_Mesh: 
                    rgbd_mesh_opt.z_factor.requires_grad = True
                    rgbd_mesh_opt.depth_transfer_function.requires_grad = True
                    optimizer    = torch.optim.Adam([rgbd_mesh_opt.z_factor, *rgbd_mesh_opt.depth_transfer_function.parameters()], lr=lr)
                else: 
                    print("error: invalid train mode for current rgb_mesh type!")
                    exit()
            elif cur_train_mode == "color":
                # optimization on color
                rgbd_mesh_opt.color_image.requires_grad = True
                optimizer    = torch.optim.Adam([rgbd_mesh_opt.color_image], lr=lr)
            elif cur_train_mode == "pos":
                # optimization on pos
                rgbd_mesh_opt.clip_pos.requires_grad = True
                optimizer    = torch.optim.Adam([rgbd_mesh_opt.clip_pos], lr=lr)
            elif cur_train_mode == "noise":
                # optimization on color
                rgbd_mesh_opt.depth_image.noise.requires_grad = True
                optimizer    = torch.optim.Adam([rgbd_mesh_opt.depth_image.noise], lr=lr)
            elif cur_train_mode == "scale": 
                # optimization of offset and scale
                if type(rgbd_mesh_opt) == Adaptive_RGBD_Mesh:
                    rgbd_mesh_opt.offset_scale.requires_grad = True
                    optimizer    = torch.optim.Adam([rgbd_mesh_opt.offset_scale], lr=lr)
                else:   
                    rgbd_mesh_opt.depth_image.offset_scale.requires_grad = True
                    optimizer    = torch.optim.Adam([rgbd_mesh_opt.depth_image.offset_scale], lr=args.lr)
                    
            return optimizer
        
        optimizer = setup_optimizer_for_train_mode()
    
        if args.use_tonemapper:
            p_grps = tonemapper.get_optimizer_param_groups(-1)
            optimizer.add_param_group(p_grps[0])
            optimizer.add_param_group(p_grps[1])
        
        img_losses = {"mse":{}} # "vgg":{},
        if args.use_feat_enc: img_losses["feat_mse"] = {} 
        start_time = time.time()
        last_write_img_it = -10*args.log_interval
        
        
        for cur_it, target in enumerate(tqdm.tqdm(dataloader_train, mininterval=args.log_tqdm_interval)):
            it = cur_it + start_it
            if type(rgbd_mesh_opt) == Adaptive_RGBD_Mesh:
                rgbd_mesh_opt.update_depth_image_from_pos()
                # if it == 0:
                    # if type(args.input_colmap_sparse) != type(None):
                    #     point_regularizer.plot(rgbd_mesh_opt.depth_image.get_absolute)
                    
            
            # batch to device
            target = {k:v.cuda() if hasattr(v,"cuda") else v for k,v in target.items()}
           
            # render
            out_attrs = ["color", "depth"]
            out = None
            if args.ref_dm_available:
                with torch.no_grad():
                    out     = rgbd_mesh_ref.render(glctx, target["mvp"], args.cam_resolution, out_attr=out_attrs, enable_mip=False, max_mip_level=0, mv=target["mv"])
            out_attrs += ["rast_out", "uv", "edge_map"]
            if args.use_feat_enc:       out_attrs += ["feat"]
            out_opt = rgbd_mesh_opt.render(glctx, target["mvp"], args.cam_resolution, out_attr=out_attrs, enable_mip=False, max_mip_level=0, mv=target["mv"])

            # tonemapper
            if cur_loss_color_weight > 0. and args.use_tonemapper:
                out_opt["color_untonemapped"] = out_opt["color"]
                out_opt["color"] = torch.empty_like(out_opt["color_untonemapped"])
                for b in range(out_opt["color"].shape[0]):
                    
                    # first bring the main cam img to hdr space
                    # we set the main camera to timestamp 0... implies that we have the params twice: :(
                    hdr_out_color = tonemapper.inverse_forward(out_opt["color_untonemapped"][b,None].permute(0,3,1,2), camera_timestamp=idx_of_view_in_dataset/(dataset_train.n_images-1), train_mode=True)
                    # secondly, bring it from hdr to the color space of the target views
                    out_opt["color"][b] = tonemapper(hdr_out_color, camera_timestamp=(target["idx"][b].item())/(dataset_train.n_images-1), train_mode=True).permute(0,2,3,1)

            # use feat enc
            if args.use_feat_enc:
                out_opt["feat_s"] = util.scale_img_nhwc(out_opt["feat"], target["feat"].shape[1:3])
                out_opt["mask_s"] = util.scale_img_nhwc(out_opt["mask"], target["feat"].shape[1:3])

            ##### Compute losses and train.
            loss = torch.zeros([1]).cuda()
            
            #### color based losses
            # mse 
            if cur_loss_color_weight > 0.0:
                color_loss_mask = out_opt["mask"]*out_opt["edge_map"] # adding mask should prevent border gradients form dr.antialias
                loss = loss + cur_loss_color_weight*mse_loss(target["img"]*color_loss_mask, out_opt["color"]*color_loss_mask)  
                loss = loss + loss*torch.mean(1-out_opt["edge_map"])
          
            # # mse on features
            if args.use_feat_enc:
                loss = loss + args.loss_feat_weight*mse_loss(target["feat"]*out_opt["mask_s"], out_opt["feat_s"])  
            
            ### regularizers
            # poisson
            p_loss, weights, grad_x, grad_y, grad_x_s, grad_y_s, = poisson_loss(depth_image_opt.get_relative,depth_image_opt.valid_mask) #depth_image_opt.valid_mask * ?

            loss = loss + args.loss_poisson_weight*p_loss
            if args.pcd_available:
                point_loss = point_regularizer(depth_image_opt.get_absolute)
                loss = loss + args.loss_point_weight*point_loss
            
                # # directly train depth transfer func on sparse pcd
                if "transfer_func" in cur_train_mode:
                    uvs = 0.5*point_regularizer.points2D_11+0.5
                    uvs[:,1] = -uvs[:,1]
                    # we use the depth values from the initial depthmap as input for the mlp
                    sampled_depth_init = point_regularizer.sample_depths(depth_image_init.get_absolute)
                    positive_tranferred_depth = rgbd_mesh_opt.depth_transfer_function(0.5*point_regularizer.points2D_11+0.5, sampled_depth_init)
                    loss = loss + args.loss_point_transfer_weight*point_regularizer.loss(positive_tranferred_depth)
                

            # median regularizer
            if args.loss_median_weight > 0.:
                filtered_dm = util.median_filter(depth_image_opt.get_absolute.unsqueeze(-1), kernel_size=5, handle_zeros=True)
                loss = loss + args.loss_median_weight*mse_loss(filtered_dm,depth_image_opt.get_absolute.unsqueeze(-1))
            
            # smoothness regularizer        
            if args.loss_smooth_weight > 0.:
                loss = loss + args.loss_smooth_weight*Loss.smooth_loss(depth_image_opt.get_absolute, depth_image_opt.valid_mask)

            # edginess of uvs regularizer
            if args.loss_uv_edginess_weight > 0.:
                _, uvs = rgbd_mesh_opt.get_pos_w_uv_offset() 
                loss = loss + args.loss_uv_edginess_weight*Loss.uv_to_edges_loss(rgbd_mesh_opt.edge_map, uvs)
                   
            # matching_edges regularizer
            cur_edge_map = None
            if args.loss_edge_map_weight > 0.:
                cur_edge_map = calc_edge_map(depth_image_opt.get_absolute.unsqueeze(0).unsqueeze(0)).permute(0,2,3,1)
                cur_edge_map = torch.clamp(cur_edge_map, -1.,1.)
                edge_map = torch.clamp(rgbd_mesh_opt.edge_map, -1.,1.)
                loss = loss + args.loss_edge_map_weight*mse_loss(  cur_edge_map, edge_map)
            
            if args.normal_available and args.loss_normal_weight > 0.:
                if cur_edge_map == None:
                    cur_edge_map = calc_edge_map(depth_image_opt.get_absolute.unsqueeze(0).unsqueeze(0)).permute(0,2,3,1)
                cur_edge_map = torch.clamp(cur_edge_map, 0.,1.)

                vpos =  rgbd_mesh_opt.render_self(out_attr=["vpos"])["vpos"]
                pred_normals = calc_normals(vpos.permute(0,3,1,2))
                loss = loss + args.loss_normal_weight*normal_regularizer(pred_normals, cur_edge_map.permute(0,3,1,2))
    
            
            if it > max_iter: 
                print("before last optim step", flush=True)
                break
            
            #### update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if it > max_iter: 
                torch.cuda.synchronize()
                print("break out of current train phase", flush=True)
                break
            
            # Print/save log.
            with torch.no_grad():
                tboard.add_scalar("loss/sum", loss, it)
                for i in range(target["mv"].shape[0]):
                    img_idx = target['idx'][i].item()
                     
                    if cur_loss_color_weight > 0.0:
                        img_losses["mse"][img_idx] = cur_loss_color_weight*mse_loss(target["img"][i]*out_opt["mask"][i], out_opt["color"][i]).item() 
                    else: img_losses["mse"][img_idx] = 0.0
                    
                    if args.use_feat_enc:
                        img_losses["feat_mse"][img_idx] = args.loss_feat_weight*mse_loss(target["feat"][i]*out_opt["mask_s"][i], out_opt["feat_s"][i]).item() 
                    
                    # print(img_idx, old_val, img_losses[img_idx])
                if args.log_interval and (it % args.log_interval == 0):
                        # calc mean/depth/regularizer losses
                        means = {k: sum(v.values())/ len(v) for k, v in img_losses.items()}  
                        
                        # pcd alignment losses
                        if args.pcd_available:
                            # direct loss
                            point_loss = point_regularizer(depth_image_opt.get_absolute).item()
                            means["point reg"] = args.loss_point_weight*point_loss
                            
                            # via mlp
                            if "transfer_func" in cur_train_mode:
                                # # directly train depth transfer func on sparse pcd
                                uvs = 0.5*point_regularizer.points2D_11+0.5
                                uvs[:,1] = -uvs[:,1]
                                sampled_depth_init = point_regularizer.sample_depths(depth_image_init.get_absolute)
                                positive_tranferred_depth = rgbd_mesh_opt.depth_transfer_function(0.5*point_regularizer.points2D_11+0.5, sampled_depth_init)
                                means["transfer pcd loss"] = args.loss_point_transfer_weight*point_regularizer.loss(positive_tranferred_depth)
                            
                            # optionally plot current state of pcd alignment
                            # if (it % (50*args.log_interval) == 0):
                            #     sampled_depth_ref = point_regularizer.sample_depths(depth_image_ref.get_absolute)
                            #     sampled_depth_init = point_regularizer.sample_depths(depth_image_init.get_absolute)
                            #     plot_depths = {"init": sampled_depth_init}
                            #     if "transfer_func" in cur_train_mode:
                            #         positive_tranferred_depth = rgbd_mesh_opt.depth_transfer_function(0.5*point_regularizer.points2D_11+0.5, sampled_depth_init)
                            #         plot_depths["mlp"] = positive_tranferred_depth
                            #     point_regularizer.plot(sampled_depth_ref, depth_image_opt.get_absolute, plot_depths ) #, args.outdir+"/depth_from_mlp.png")
                        
                        # poisson regularizer
                        p_loss, weights,grad_blended_x, grad_blended_y, grad_source_x, grad_source_y = poisson_loss(depth_image_opt.get_relative,depth_image_opt.valid_mask)
                        means["poisson"] = args.loss_poisson_weight*p_loss.item() 
                        # median regularizer                        
                        filtered_dm = util.median_filter(depth_image_opt.get_absolute.unsqueeze(-1), kernel_size=5, handle_zeros=True)
                        means["median reg"] =  args.loss_median_weight*mse_loss(filtered_dm,depth_image_opt.get_absolute.unsqueeze(-1)).item()
                        # smooth regularizer                        
                        means["smooth reg"] =  args.loss_smooth_weight*Loss.smooth_loss(depth_image_opt.get_absolute, depth_image_opt.valid_mask).item()
                        # edge_map regularizer
                        cur_edge_map = calc_edge_map(depth_image_opt.get_absolute.unsqueeze(0).unsqueeze(0)).permute(0,2,3,1)
                        cur_edge_map = torch.clamp(cur_edge_map, -1.,1.)
                        edge_map = torch.clamp(rgbd_mesh_opt.edge_map, -1.,1.)
                        means["edge map reg"] = args.loss_edge_map_weight*mse_loss(  cur_edge_map, edge_map).item()
                        # uv egdiness regularizer
                        _, uvs = rgbd_mesh_opt.get_pos_w_uv_offset() 
                        means["uv edginess"] = args.loss_uv_edginess_weight*Loss.uv_to_edges_loss(rgbd_mesh_opt.edge_map, uvs).item()
                        # normals
                        if args.normal_available:
                            vpos =  rgbd_mesh_opt.render_self(out_attr=["vpos"])["vpos"]
                            pred_normals = calc_normals(vpos.permute(0,3,1,2))
                            cur_edge_map = torch.clamp(cur_edge_map, 0.,1.)
                            means["normal"] =  args.loss_normal_weight*normal_regularizer(pred_normals, cur_edge_map.permute(0,3,1,2))
                        
                    
                        # calc metrics by comparing agains referencs depth
                        metrics = {}
                        if args.ref_dm_available:
                            metrics = calc_metrics(args, depth_image_opt.get_absolute, depth_image_ref.get_absolute, depth_image_ref.valid_mask)
                        
                        # log file
                        cur_t = time.time() - start_time  
                        s = f"t={util.time_to_text(cur_t)}, iter={it}, num={len(img_losses['mse'])}, err={str({**means, **metrics})}\n"
                        s += f"dm min: {depth_image_opt.absmin:.06f}/{depth_image_opt.relmin:.06f}, max: {depth_image_opt.absmax:.06f}/{depth_image_opt.relmax:.06f}, offset_scale_dm: {depth_image_opt.offset_scale}, offset_scale_mesh: {rgbd_mesh_opt.offset_scale}"
                        if args.use_tonemapper:
                            s += f"\n exposure params: {tonemapper.exposure_params}\n"
                            np.savetxt(args.outdir+"/exposure.txt", tonemapper.exposure_params.clone().detach().cpu().numpy())
                            np.savetxt(args.outdir+"/response.txt", tonemapper.response_params.clone().detach().squeeze().cpu().numpy())
                            
                        if (it % (10*args.log_interval) == 0):
                            print(s)
                        if log_file:
                            log_file.write(s + "\n")
                            log_file.flush()
                            
                        # tboard
                        for k, v in means.items(): tboard.add_scalar("loss/"+k, v, it)
                        tboard.add_scalar("loss/sum", loss, it)
                        log_metrics_with_args(tboard, it, args.log_hparams, args, metrics )
                        
                        # depth Image output
                        if (it % (10*args.log_interval) == 0):
                            write_depthmap_output(args, it, current_stage_index, rgbd_mesh_opt, rgbd_mesh_ref, tboard)
                        if (it % (50*args.log_interval) == 0):
                            print("Updated pcd in tboard")
                            point_size_config = {
                                'cls': 'PointsMaterial',
                                'size': 10
                            }


                # Show/save image.
                b = -1
                for i in range(target["mv"].shape[0]): 
                    # if target['idx'][i].item() == 4 % dataset_train.n_images: b = i
                    if target['idx'][i].item() == 19 % dataset_train.n_images: b = i
                if b != -1: # display_image or save_mp4:
                    if args.log_optim_anim and args.ref_dm_available:
                        write_anim_output(args, it, b, out, out_opt, target)
                        write_anim_depthmap_output(args, it, current_stage_index, rgbd_mesh_opt, rgbd_mesh_ref, glctx)
                    if it - last_write_img_it > 10*args.log_interval: 
                        write_img_output(args, it, b, out, out_opt, target, tboard)
                        last_write_img_it = it

        return it

    # Multi Stage Training

    last_it = 0
    # first phase: do coarse aligment via MLP and align uvs to edges  (args.init_train_mode == "transfer_func_uv")
    if args.init_max_iter > 0:        
        last_it = train(last_it, 1, args.init_max_iter, args.init_train_mode, args.init_lr, args.init_loss_color_weight)   

    # bake previous results and reset train mode etc.   
    print(50*"#")
    print("bake mesh after mlp", flush=True)
    rgbd_mesh_opt.bake_init_transfer_func_uv_alignment()
    # currently, the maximum range a vertex can move in uv space is 0.25*meshdownscale[px],
    # after baking we thus allow another 0.25 *meshdownscale[px] movement
    rgbd_mesh_opt.set_train_mode(args, "train_"+args.train_mode)
    rgbd_mesh_opt.simplifiy_mesh(args.adaptive_mesh_simplification_ratio)
    
    # second phase: do vertex based refinement (args.train_mode == "depht_uv")
    last_it = train(last_it, 2, args.max_iter, args.train_mode, args.lr, args.loss_color_weight)                    

    # Done.
    for writer in writers.values():
        writer.close()
    if log_file:
        log_file.close()
    if tboard is not None: 
        tboard.close()

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Depth map fitting')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('--base_dir', help='specify base input directory')
    parser.add_argument('--input_dir', help='specify input directory')
    parser.add_argument('--input_colmap_sparse' , help='specify input directory')
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--view_name', help='specify current view', default='')
    parser.add_argument('--input_normal_dir', help='specify directory for normal maps as regularizers')

    
    parser.add_argument("--sparse_pcd_from_ref_outlier_ratio",  type=float,     default=0.02)
    parser.add_argument("--sparse_pcd_from_ref_num_points",     type=int,       default=5000)
    parser.add_argument("--sparse_pcd_from_ref_perturb_std",    type=float,       default=0.02)
    
    
    parser.add_argument("--lr",         type=float)
    parser.add_argument('--max_iter',   type=int,   default=24000)
    parser.add_argument('--batch',      type=int,   default=4)
    parser.add_argument('--use_opengl', action='store_true', default=False)
    parser.add_argument('--load_mode',  type=str,   default="lazy_load")
    parser.add_argument('--srgb', action='store_true', default=False)
    
    parser.add_argument("--log_interval",               type=int, default=10)
    parser.add_argument("--log_tqdm_interval",          type=float, default=1.0)
    parser.add_argument("--log_hparams",                type=str,    nargs='+')
    parser.add_argument("--log_optim_anim",             type=int,     default=0)

    parser.add_argument('--use_tonemapper',     action='store_true', default=False)
    parser.add_argument('--use_feat_enc',       action='store_true', default=False)

    parser.add_argument('--train_mode',     help='specify what is optmized', default="depth_uv")
    
    
    parser.add_argument("--scale_init_lr",       type=float)
    parser.add_argument('--init_train_mode',     help='specify what is optmized', default="transfer_func_uv")
    parser.add_argument('--init_max_iter',   type=int,   default=400)
    parser.add_argument("--init_lr",         type=float)
    
    parser.add_argument("--optim_stage_starts",     type=int,    nargs='+',    default=[0, 150, 300, 800])
    parser.add_argument("--optim_stage_lr_factor",  type=float,  nargs='+',    default=[ 2, 1.5, 1.25, 1.])
    
    parser.add_argument('--cam_resolution', type=int,   default=[720,1280],   required=False)
    parser.add_argument('--cam_near_far',   type=float, default=[0.1, 100.0])    
    parser.add_argument('--cam_use_flip_mat',     type=int,     default=0, help='use for replica, tnt transform files, and do not use for scannet!')
    
    parser.add_argument('--init_depth_dir', help='specify directory with init depth maps', default="depth_marigold")
    parser.add_argument('--init_ref_depth_dir', help='specify directory with reference depth maps (for evaluation)', default="depth")
    parser.add_argument('--init_depth_mode', help='specify which kind of depth map is provided, valid: absolute, relative_to_absdm, relative_to_abspcd', default="relative_to_abspcd")
    parser.add_argument('--init_adaptive_mesh_simplification_ratio',   type=float,     default=1.)
    
    parser.add_argument('--depth_image_layers_from_zeros',   action='store_true', default=False)
    parser.add_argument("--depth_image_downsample", type=int,       default=4)
    parser.add_argument("--depth_image_depth_scale", type=float,       default=0.001)
    
    parser.add_argument('--adaptive_mesh_simplification_ratio',   type=float,     default=0.5)
    parser.add_argument('--transfer_func_acti',                   type=str,       default="relu")
    parser.add_argument('--transfer_func_normalize_z',            type=int,       default=True)
    parser.add_argument('--transfer_func_hlayers',                type=int,    nargs='+',       default=[16,16])
    parser.add_argument('--transfer_func_posencs_uv_z',           type=int,    nargs='+',       default=[3,5])
    parser.add_argument('--transfer_func_init_std',               type=float,     default=0.1)
    
    parser.add_argument('--scale_init_align_for',   type=str,     default="median")
    parser.add_argument('--scale_init_percentile',   type=float,     default=0.001)
    
    parser.add_argument('--scale_init_loss_color_weight',   type=float,     default=0.0)
    parser.add_argument('--init_loss_color_weight',         type=float,     default=0.0)
    parser.add_argument('--loss_color_weight',              type=float,     default=1.0)
    parser.add_argument('--loss_poisson_weight',            type=float,     default=400.0)
    parser.add_argument('--loss_poisson_blur_window',       type=int,       default=5)
    parser.add_argument('--loss_feat_weight',               type=float,     default=0.00)
    parser.add_argument('--loss_point_weight',               type=float,     default=0.1)
    parser.add_argument('--loss_point_transfer_weight',      type=float,     default=1.0)
    parser.add_argument('--loss_edge_map_weight',            type=float,     default=0.1)
    parser.add_argument('--loss_uv_edginess_weight',         type=float,     default=0.1)
    parser.add_argument('--loss_median_weight',             type=float,     default=0.0)
    parser.add_argument('--loss_smooth_weight',             type=float,     default=.00)
    parser.add_argument('--loss_normal_weight',             type=float,     default=.01)
    
    
    args = parser.parse_args()

    if args.config is not None:
        data = json.load(open(args.config, 'r'))
        for key in data:
            if not key in args.__dict__ or args.__dict__[key] == parser.get_default(key): 
                args.__dict__[key] = data[key]
    print(10*"#")
    print(args)
    print(10*"#")
    if not args.input_dir:
        print("error: no input dir given!")
        exit()
    if not args.log_hparams:
        print("error: no hparams given!")
        exit()
        
    if args.base_dir:
        print("RefineDepth: Concat base and input dir to")
        args.__dict__["input_dir"] = args.__dict__["base_dir"] + "/"+ args.__dict__["input_dir"]
        print(args.input_dir)
        if args.input_colmap_sparse != "None" and type(args.input_colmap_sparse) != type(None):
            args.__dict__["input_colmap_sparse"] = args.__dict__["base_dir"] + "/"+ args.__dict__["input_colmap_sparse"]
            args.__dict__["pcd_available"] = True 
        else:             
            args.__dict__["pcd_available"] = False
              
        if args.input_normal_dir != "None" and type(args.input_normal_dir) != type(None):
            args.__dict__["input_normal_dir"] = args.__dict__["base_dir"] + "/"+ args.__dict__["input_normal_dir"]
            args.__dict__["normal_available"] = True 
        else:             
            args.__dict__["normal_available"] = False  
            
        if args.init_ref_depth_dir != "None" and type(args.init_ref_depth_dir) != type(None):
            args.__dict__["ref_dm_available"] = True 
        else:             
            args.__dict__["ref_dm_available"] = False  

    # Set up logging.
    if args.outdir:
        print (f'Saving results under {args.outdir}')
   
        # Run.
    # while True:
    #     oom = False
    #     try:
    #         refine_depthmap(
    #             args=args,
    #             log_fn='log.txt',
    #             mp4save_fn='progress.mp4',
    #         )
    #     except RuntimeError as e:  # Out of memory
    #         print(repr(e))
    #         oom = True
    #         args.__dict__["batch"] -= 4
    #         print("set batch size to", args.batch)
    #     if not oom: break

    refine_depthmap(
                args=args,
                log_fn='log.txt',
                mp4save_fn='progress.mp4',
            )

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
