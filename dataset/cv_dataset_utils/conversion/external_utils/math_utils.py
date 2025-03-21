import numpy as np
import math


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

def inv_R_T(R,t):
    new_R = R.T
    new_t = -new_R @ t
    return new_R, new_t

def fov_from_f_radians(f, pixels):
    angle = math.atan(pixels / (f * 2)) * 2
    return angle

def fov_from_f(f, pixels):
    return fov_from_f_radians(f,pixels) * 180 / math.pi


def rotmat_from_vec3_to_vec3(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat_from_vec3_to_vec3(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom