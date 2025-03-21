from pathlib import Path

try:
    from  scene import *
    import external_utils.colmap_utils as colmap_utils
except:
    from .scene import *
    from .external_utils import colmap_utils
    

def load_colmap_as_scene(input_dir) -> BaseScene:
    scene = BaseScene()
    input_dir = Path(input_dir)
    
    # load cameras
    if (input_dir / "cameras.bin").exists():
        colmap_cams = colmap_utils.read_cameras_binary(input_dir / "cameras.bin")
    elif (input_dir / "cameras.txt").exists():
        colmap_cams = colmap_utils.read_cameras_text(input_dir / "cameras.txt")
    else: 
        print("error: no camera files found in", input_dir)
        exit()

    for colmap_cam in colmap_cams.values():
        camera = None
        if colmap_cam.model == "SIMPLE_PINHOLE":
            camera = BaseCamera(colmap_cam.id, 
                        fx=colmap_cam.params[0], fy=colmap_cam.params[0], 
                        cx=colmap_cam.params[1], cy=colmap_cam.params[2], 
                        width=colmap_cam.width, height=colmap_cam.height)
        if colmap_cam.model == "PINHOLE":
            camera = BaseCamera(colmap_cam.id, 
                        fx=colmap_cam.params[0], fy=colmap_cam.params[1], 
                        cx=colmap_cam.params[2], cy=colmap_cam.params[3], 
                        width=colmap_cam.width, height=colmap_cam.height)
        if camera == None:
            camera = BaseCamera(colmap_cam.id, 
                        fx=colmap_cam.params[0], fy=colmap_cam.params[1], 
                        cx=colmap_cam.params[2], cy=colmap_cam.params[3], 
                        width=colmap_cam.width, height=colmap_cam.height)
            print(f"error: {colmap_cam.model} is an unsupported cam type!")
            
        scene.cameras[camera.id] = camera
    
        
    # load views 
    if (input_dir / "images.bin").exists():
        colmap_ims = colmap_utils.read_images_binary(input_dir / "images.bin")
    else:
        colmap_ims = colmap_utils.read_images_text(input_dir / "images.txt")
    for colmap_im in colmap_ims.values():
        R = colmap_utils.rotm_from_quat(colmap_im.qvec)
        T = colmap_im.tvec
        camera = scene.cameras[colmap_im.camera_id]
        image_name = Path(colmap_im.name).name
        scene.color_path = Path(colmap_im.name).parent # WARNING: assumes only one path, if there are different one it'll break
        scene.views[colmap_im.id] = BaseView(colmap_im.id, R=R, T=T, camera=camera, image_name=image_name)


    # load points?
    # TODO

    return scene


def write_scene_as_colmap(scene : BaseScene, output_dir, format=".txt", sort_filenames=True):
    if sort_filenames:
        views  = dict(sorted({str(Path(v.image_name).stem): v for v in scene.views.values()}.items()))
    else:
        views = scene.views
    output_dir = Path(output_dir)
    colmap_cams = {}
    for camera in scene.cameras.values():
        colmap_cams[camera.id] = colmap_utils.Camera(camera.id, model="PINHOLE", # we assume that we only handle pinhole cameras! 
                                         width=camera.width, height=camera.height, 
                                         params=[camera.fx, camera.fy, camera.cx, camera.cy])
    colmap_images = {}
    for view in views.values():
        colmap_images[view.id] = colmap_utils.Image(view.id, colmap_utils.quat_from_rotm(view.R), view.T, view.camera.id, str(scene.color_path/view.image_name), 
                                           [], []) # TODO: point3Ds needed?
        
    if format == ".txt":
        colmap_utils.write_cameras_text(out_path=output_dir/"cameras.txt", cameras=colmap_cams)
        colmap_utils.write_images_text(out_path=output_dir/"images.txt", images=colmap_images)
        
    elif format == ".bin":
        colmap_utils.write_cameras_text(out_path=output_dir/"cameras.bin", cameras=colmap_cams)
        colmap_utils.write_images_text(out_path=output_dir/"images.bin", images=colmap_images)
        
    else:
        print(f"error: {format} invalid format!")
        
    