
import numpy as np
import argparse
import struct

import imageio
def read_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)

def read_depth_image(fn, read_depth_scale=0.001) -> np.ndarray:
    return (read_depth_scale*read_image_raw(fn).astype(np.float64)).astype(np.float32)

# from PIL import Image
# def read_depth_image(fn):
#     return np.array(Image.open(fn), dtype='u2')

def read_depth_colmap(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    depth_map = np.transpose(array, (1, 0, 2)).squeeze()
    min_depth, max_depth = np.percentile(
    depth_map, [args.min_depth_percentile, args.max_depth_percentile]
)
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    
    return depth_map


def write_depth_colmap(path, array):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list
        )
        fid.write(byte_data)



import cv2
def write_depth_image(fn, x : np.array, write_depth_scale=1000.0):
    # np_depth = np.clip(np_depth, 0, 100) # for now clamp to 100m
    x = write_depth_scale * x
    # Save depth images (convert to uint16)
    cv2.imwrite(str(fn), x.astype(np.uint16))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_dm",
        type=str,
        required=True
    )
    parser.add_argument(
        "--input_format",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dm",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_format",
        type=str,
        required=True
    )
    parser.add_argument(
        "--min_depth_percentile",
        help="minimum visualization depth percentile",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--max_depth_percentile",
        help="maximum visualization depth percentile",
        type=float,
        default=95,
    )
    parser.add_argument(
        "--input_depth_scale",
        help="depth scale applied when loaded form e.g. pngs",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--output_depth_scale",
        help="depth scale applied when written form e.g. pngs",
        type=float,
        default=1000.0,
    )
    
    args = parser.parse_args()
    
    print(args)
    
    if args.input_format == "image":
        dm = read_depth_image(args.input_dm, args.input_depth_scale)
    elif args.input_format == "colmap":
        dm = read_depth_colmap(args.input_dm)
    else:
        print("input_format: ", args.input_format, "  NO NO")
        exit()
    
    print(dm.max(), dm.min(), dm.mean())
    
    if args.output_format == "image":
        write_depth_image(args.output_dm, dm, args.output_depth_scale)
    elif args.output_format == "colmap":
        write_depth_colmap(args.output_dm, dm)
    else:
        print("output_format: ", args.output_format, "  NO NO")
        exit()