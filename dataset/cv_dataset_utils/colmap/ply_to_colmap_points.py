import plyio

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

if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_dir = sys.argv[2]
    format = sys.argv[3]
    
    
    pc = plyio.read_ply(input_filename)["points"]
    
    # print(pc)
    colmap_points = {}
    for i, (x,y,z,r,g,b) in enumerate(zip(pc.x, pc.y, pc.z, pc.red, pc.green, pc.blue)):
        xyz = np.array([x,y,z])
        rgb = np.array([r,g,b])
        
        colmap_points[i] = colmap_utils.Point3D(
                                id=i,
                                xyz=xyz,
                                rgb=rgb,
                                error=0.0,
                                image_ids=np.array([]),
                                point2D_idxs=np.array([]),
                            )
    if format == "bin":
        colmap_utils.write_points3D_binary(output_dir+"points3D.bin", colmap_points)
    elif format == "txt":
        colmap_utils.write_points3D_text(output_dir+"points3D.txt", colmap_points)
    else:
        print(f"{format} is invalid")
    

    
    # point3D_id = binary_point_line_properties[0]
    # xyz = np.array(binary_point_line_properties[1:4])
    # rgb = np.array(binary_point_line_properties[4:7])
    # error = np.array(binary_point_line_properties[7])
    # track_length = read_next_bytes(
    #     fid, num_bytes=8, format_char_sequence="Q"
    # )[0]
    # track_elems = read_next_bytes(
    #     fid,
    #     num_bytes=8 * track_length,
    #     format_char_sequence="ii" * track_length,
    # )
    # image_ids = np.array(tuple(map(int, track_elems[0::2])))
    # point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
    # points3D[point3D_id] = Point3D(
    #     id=point3D_id,
    #     xyz=xyz,
    #     rgb=rgb,
    #     error=error,
    #     image_ids=image_ids,
    #     point2D_idxs=point2D_idxs,
    # )
    