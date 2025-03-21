import sys
try:
    sys.path.append("..")
    from conversion.scene import *
    import conversion.colmap  as colmap
    import conversion.external_utils.colmap_utils  as colmap_utils
except:
    from ..conversion.scene import *
    from ..conversion import colmap
    from ..conversion.external_utils import colmap_utils

    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_plot2D(data):
    fig = plt.figure()

    # Create scatter plot
    plt.scatter(*[data[:,i] for i in range(data.shape[1])])  # data[:, 0] is x, data[:, 1] is y

    # Optional: Set labels and title
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('2D Scatter Plot')

    # Show plot
    plt.show()

def scatter_plot3D(data, color=None):
    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # # Add colors
    # color_config = {}
    # if type(color) != type(None):
    #     color_config = {"mode":'markers', "marker": dict(color=[f"rgb({c[0]}, {c[1]}, {c[2]})" for c in color])}
    # colors = dict(color=[f"rgb({c[0]}, {c[1]}, {c[2]})" for c in color])
    # print(colors)

    # Create scatter plot
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], **color_config)  # data[:, 0] is x, data[:, 1] is y, data[:, 2] is z
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], facecolors=color/255.0) #mode='markers', marker=dict(color=[f"rgb({c[0]}, {c[1]}, {c[2]})" for c in color] ))  # data[:, 0] is x, data[:, 1] is y, data[:, 2] is z

    # Optional: Set labels and title
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    ax.set_title('3D Scatter Plot')
    
    

    # Show plot
    plt.show()
    

def get_points2D_and_3D_for_img(img_id, points3d, images, width, height):
    valid_points2Ds = []
    valid_points3Ds = []
    valid_col3Ds = []
    for i, (p2D, p3D_idx) in enumerate(zip(images[img_id].xys, images[img_id].point3D_ids)):
        if p3D_idx == -1 : continue
        if np.any(p2D < 0.0) or p2D[0] > width or p2D[1] > height : continue
        
        p3D = points3d[p3D_idx].xyz
        col3D = points3d[p3D_idx].rgb
        # print(i, p2D, p3D_idx)
        valid_points2Ds.append(p2D)
        valid_points3Ds.append(p3D)
        valid_col3Ds.append(col3D)
        

    # Example data: numpy array with shape (n_points, 2)
    # Each row represents an (x, y) coordinate
    data2d = np.array(valid_points2Ds)
    data3d = np.array(valid_points3Ds)
    color3d = np.array(valid_col3Ds)
    return data2d, data3d, color3d
    
    

if __name__ == '__main__':
    colmap_dir = Path(sys.argv[1])
    # output_dir = sys.argv[2]
    # format = sys.argv[3]
    
    
    points3d = colmap_utils.read_points3d_binary(colmap_dir/"points3D.bin")
    images = colmap_utils.read_images_binary(colmap_dir/"images.bin")
    cameras = colmap_utils.read_cameras_binary(colmap_dir/"cameras.bin")
    print(list(cameras.values())[0].width, list(cameras.values())[0].height)
    # scene = colmap.load_colmap_as_scene(colmap_dir)
    # print(images)
    print(images[1].name, images[1].xys, images[1].point3D_ids)
    
    img_id = 1
    data2d, data3d, color3d = get_points2D_and_3D_for_img(img_id, points3d, images, list(cameras.values())[0].width, list(cameras.values())[0].height)
    
    #flip Y
    data2d[:,1] = -data2d[:,1] 
    # scatter_plot2D(data2d)
    scatter_plot3D(data3d, color3d)