import numpy as np
from matplotlib import colormaps
import rerun as rr
from rerun.components import Material
from scipy.spatial import transform


def color_fn(x, cmap="tab10"):
    return colormaps[cmap](x % colormaps[cmap].N)


def et_log_sample(
    root_name: str,
    traj: np.ndarray,
    char_traj: np.ndarray,
    K: np.ndarray,
):
    num_cameras = traj.shape[0]

    rr.log(root_name, rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    rr.log(
        f"{root_name}/trajectory/points",
        rr.Points3D(traj[:, :3, 3]),
        timeless=True,
    )
    rr.log(
        f"{root_name}/trajectory/line",
        rr.LineStrips3D(
            np.stack((traj[:, :3, 3][:-1], traj[:, :3, 3][1:]), axis=1),
            colors=[(1.0, 0.0, 1.0, 1.0)],
        ),
        timeless=True,
    )
    for k in range(num_cameras):
        rr.set_time_sequence("frame_idx", k)

        translation = traj[k][:3, 3]
        rotation_q = transform.Rotation.from_matrix(traj[k][:3, :3]).as_quat()
        rr.log(
            f"{root_name}/camera/image",
            rr.Pinhole(
                image_from_camera=K,
                width=K[0, -1] * 2,
                height=K[1, -1] * 2,
            ),
        )
        rr.log(
            f"{root_name}/camera",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_q),
            ),
        )
        rr.set_time_sequence("image", k)

        # Log character trajectory points
        rr.log(
            f"{root_name}/char_traj/points",
            rr.Points3D(char_traj.T, colors=[(1.0, 0.0, 0.0, 1.0)]),
            timeless=True,
        )
