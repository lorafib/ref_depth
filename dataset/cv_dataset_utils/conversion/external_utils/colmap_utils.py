# This file is from 
# https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
# To this end, we include the following copyright notice:
#
# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import numpy as np
from pathlib import Path
import sys
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                if len(elems) % 3 != 0: print(image_id, image_name, ": parsing error!")
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def array_to_blob(array):
    if sys.version_info[0] >= 3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if sys.version_info[0] >= 3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def read_array(path):
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
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(p, arr):
    arr
    shape_str = [f"{s}&" for s in [arr.shape[1], arr.shape[0], *arr.shape[2:]]]
    while len(shape_str) < 3:
        shape_str.append("1&")
    shape_str = "".join(shape_str)
    with open(p, "wb") as fp:
        fp.write(shape_str.encode("latin1"))
        arr.tofile(fp)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def write_cameras_text(out_path, cameras):
    with open(out_path, "w") as fp:
        for id, camera in cameras.items():
            params = [str(v) for v in camera.params]
            fp.write(
                f'{camera.id} {camera.model} {camera.width} {camera.height} {" ".join(params)}\n'
            )


def write_cameras_binary(out_path, cameras):
    with open(out_path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def write_images_text(out_path, images, image_callback=None):
    with open(out_path, "w") as fp:
        for id, image in images.items():
            if image_callback is not None:
                image = image_callback(image)
            fp.write(
                f"{image.id} {image.qvec[0]} {image.qvec[1]} {image.qvec[2]} {image.qvec[3]} {image.tvec[0]} {image.tvec[1]} {image.tvec[2]} {image.camera_id} {image.name}\n"
            )
            for xy, pid in zip(image.xys, image.point3D_ids):
                fp.write(f"{xy[0]} {xy[1]} {pid} ")
            fp.write("\n")


def write_images_binary(out_path, images):
    with open(out_path, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def write_points3D_text(out_path, points3D):
    with open(out_path, "w") as fp:
        for id, point in points3D.items():
            fp.write(
                f"{point.id} {point.xyz[0]} {point.xyz[1]} {point.xyz[2]} {point.rgb[0]} {point.rgb[1]} {point.rgb[2]} {point.error}"
            )
            for image_id, point2d_idx in zip(
                point.image_ids, point.point2D_idxs
            ):
                fp.write(f" {image_id} {point2d_idx}")
            fp.write("\n")


def write_points3D_binary(out_path, points3D):
    with open(out_path, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def rotm_from_quat(q):
    q = q.reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * z * w,
                2 * x * z + 2 * y * w,
            ],
            [
                2 * x * y + 2 * z * w,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * x * w,
            ],
            [
                2 * x * z - 2 * y * w,
                2 * y * z + 2 * x * w,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=q.dtype,
    )
    R = R.transpose((2, 0, 1))
    return R.squeeze()


def quat_from_rotm(R):
    # # http://muri.materials.cmu.edu/wp-content/uploads/2015/06/RotationPaperRevised.pdf
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # http://www.iri.upc.edu/files/scidoc/2083-A-Survey-on-the-Computation-of-Quaternions-from-Rotation-Matrices.pdf
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    R = R.reshape(-1, 3, 3)
    w = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]))
    x = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    y = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    z = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))
    q0 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q0[:, 0] = w
    q0[:, 1] = x * np.sign(x * (R[:, 2, 1] - R[:, 1, 2]))
    q0[:, 2] = y * np.sign(y * (R[:, 0, 2] - R[:, 2, 0]))
    q0[:, 3] = z * np.sign(z * (R[:, 1, 0] - R[:, 0, 1]))
    q1 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q1[:, 0] = w * np.sign(w * (R[:, 2, 1] - R[:, 1, 2]))
    q1[:, 1] = x
    q1[:, 2] = y * np.sign(y * (R[:, 1, 0] + R[:, 0, 1]))
    q1[:, 3] = z * np.sign(z * (R[:, 0, 2] + R[:, 2, 0]))
    q2 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q2[:, 0] = w * np.sign(w * (R[:, 0, 2] - R[:, 2, 0]))
    q2[:, 1] = x * np.sign(x * (R[:, 0, 1] + R[:, 1, 0]))
    q2[:, 2] = y
    q2[:, 3] = z * np.sign(z * (R[:, 1, 2] + R[:, 2, 1]))
    q3 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q3[:, 0] = w * np.sign(w * (R[:, 1, 0] - R[:, 0, 1]))
    q3[:, 1] = x * np.sign(x * (R[:, 0, 2] + R[:, 2, 0]))
    q3[:, 2] = y * np.sign(y * (R[:, 1, 2] + R[:, 2, 1]))
    q3[:, 3] = z
    q = q0 * (w[:, None] > 0) + (w[:, None] == 0) * (
        q1 * (x[:, None] > 0)
        + (x[:, None] == 0) * (q2 * (y[:, None] > 0) + (y[:, None] == 0) * (q3))
    )
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.squeeze()



def load_cameras(sparse_dir, im_paths):
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "images.bin").exists():
        ims = read_images_binary(sparse_dir / "images.bin")
    else:
        ims = read_images_text(sparse_dir / "images.txt")
    if (sparse_dir / "cameras.bin").exists():
        cams = read_cameras_binary(sparse_dir / "cameras.bin")
    else:
        cams = read_cameras_text(sparse_dir / "cameras.txt")

    ims = {im.name: im for key, im in ims.items()}

    Ks = np.empty((len(im_paths), 3, 3), dtype=np.float64)
    Rs = np.empty((len(im_paths), 3, 3), dtype=np.float64)
    ts = np.empty((len(im_paths), 3), dtype=np.float64)
    for idx, im_path in enumerate(im_paths):
        im = ims[im_path.name]
        camera_id = im.camera_id
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cams[camera_id].params
        Ks[idx] = K
        Rs[idx] = rotm_from_quat(im.qvec)
        ts[idx] = im.tvec

    return Ks, Rs, ts

def load_cameras_all(sparse_dir):
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "images.bin").exists():
        ims = read_images_binary(sparse_dir / "images.bin")
    else:
        ims = read_images_text(sparse_dir / "images.txt")
    if (sparse_dir / "cameras.bin").exists():
        cams = read_cameras_binary(sparse_dir / "cameras.bin")
    else:
        cams = read_cameras_text(sparse_dir / "cameras.txt")

    ims = {im.name: im for key, im in ims.items()}

    Ks = []
    Rs = []
    ts = []
    heights, widths = [], []
    for im_name in ims.keys():
        im = ims[im_name]
        camera_id = im.camera_id
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cams[camera_id].params
        Ks.append(K)
        Rs.append(rotm_from_quat(im.qvec))
        ts.append(im.tvec)
        heights.append(cams[camera_id].height)
        widths.append(cams[camera_id].width)

    Ks = np.array(Ks)
    Rs = np.array(Rs)
    ts = np.array(ts)
    heights = np.array(heights)
    widths = np.array(widths)

    return Ks, Rs, ts, heights, widths