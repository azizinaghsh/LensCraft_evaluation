import numpy as np
import torch
from scipy.spatial.transform import Rotation as transform

def ccd_transform_to_7DoF(traj):
    traj = traj[0]
    num_cameras = traj.shape[0]

    camera_data = []
    target_num_cameras = 30

    if num_cameras > target_num_cameras:
        downsampled_traj_indices = np.linspace(0, num_cameras - 1, target_num_cameras, dtype=int)
        downsampled_traj = traj[downsampled_traj_indices]
    elif num_cameras < target_num_cameras:
        upsampled_traj_indices = np.linspace(0, num_cameras - 1, target_num_cameras)
        upsampled_traj = np.array([np.interp(upsampled_traj_indices, np.arange(num_cameras), traj[:, i]) for i in range(traj.shape[1])]).T
        downsampled_traj = upsampled_traj
    else:
        downsampled_traj = traj

    for k in range(downsampled_traj.shape[0]):
        translation = downsampled_traj[k][:3]

        fov = 60.0
        aspect_ratio = 16 / 9

        fov_x = np.deg2rad(fov)
        fov_y = 2 * np.arctan(np.tan(fov_x / 2) / aspect_ratio)

        horizontal_angle = np.arctan(downsampled_traj[k][3] * np.tan(fov_x / 2))
        vertical_angle = np.arctan(downsampled_traj[k][4] * np.tan(fov_y / 2))

        direction = -translation
        direction /= np.linalg.norm(direction)

        up = np.array([0, 1, 0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)

        rotation_matrix = np.vstack([right, up, direction]).T

        rotation_x = transform.Rotation.from_rotvec(vertical_angle * np.array([1, 0, 0]))
        rotation_y = transform.Rotation.from_rotvec(-horizontal_angle * np.array([0, 1, 0]))

        rotation_combined = rotation_y * transform.Rotation.from_matrix(rotation_matrix) * rotation_x

        euler_angles = rotation_combined.as_euler('xyz', degrees=True)

        focal_length = 955.02
        camera_tensor = np.hstack([translation, euler_angles, [focal_length]])

        camera_data.append(camera_tensor)

    return torch.tensor(camera_data, dtype=torch.float)
