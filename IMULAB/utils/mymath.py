import numpy as np

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1],  v[0],  0],
    ])


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector → unit quaternion [w, x, y, z]."""
    angle = np.linalg.norm(v)
    if angle < 1e-8:                          # first-order approx avoids /0
        return np.array([1.0, v[0]/2, v[1]/2, v[2]/2])
    axis = v / angle
    return np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])


def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w,x,y,z] → 3×3 rotation matrix (body→world)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])