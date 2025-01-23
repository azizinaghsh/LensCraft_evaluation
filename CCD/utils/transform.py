import numpy as np
import torch
from scipy.spatial.transform import Rotation as transform

def ccd_transform_to_7DoF(traj):
    """
    Convert an input trajectory into a 7DoF camera representation.

    The input 'traj' is expected to be of shape (1, N, 5), where:
      - The first dimension is usually just batch-like (so we do traj = traj[0]).
      - The second dimension (N) is the number of camera frames.
      - The third dimension (5) should contain [x, y, z, horizontal_factor, vertical_factor].
    """

    # Extract the first element if traj is shaped (1, N, 5)
    traj = traj[0]
    num_cameras = traj.shape[0]

    camera_data = []
    target_num_cameras = 30

    # Downsample or upsample to ensure we have exactly target_num_cameras frames
    if num_cameras > target_num_cameras:
        # Downsample
        downsampled_traj_indices = np.linspace(0, num_cameras - 1, target_num_cameras, dtype=int)
        downsampled_traj = traj[downsampled_traj_indices]
    elif num_cameras < target_num_cameras:
        # Upsample
        upsampled_traj_indices = np.linspace(0, num_cameras - 1, target_num_cameras)
        # Interpolate each column separately
        upsampled_traj = np.array([
            np.interp(upsampled_traj_indices, np.arange(num_cameras), traj[:, i])
            for i in range(traj.shape[1])
        ]).T
        downsampled_traj = upsampled_traj
    else:
        downsampled_traj = traj

    # Loop over each frame in the (down/upsampled) trajectory
    for k in range(downsampled_traj.shape[0]):
        # The first 3 values are translation
        translation = downsampled_traj[k][:3]

        # The next 2 values affect horizontal/vertical angles within FOV
        # (Assumed to be "horizontal_factor" and "vertical_factor" in some range)
        horizontal_factor = downsampled_traj[k][3]
        vertical_factor   = downsampled_traj[k][4]

        # Fixed FoV and aspect ratio
        fov = 60.0
        aspect_ratio = 16 / 9

        # Convert FoV from degrees to radians
        fov_x = np.deg2rad(fov)
        fov_y = 2.0 * np.arctan(np.tan(fov_x / 2.0) / aspect_ratio)

        # Compute angles
        horizontal_angle = np.arctan(horizontal_factor * np.tan(fov_x / 2.0))
        vertical_angle   = np.arctan(vertical_factor   * np.tan(fov_y / 2.0))

        # Look-at direction set to the negative of translation (pointing towards origin)
        direction = -translation
        direction /= np.linalg.norm(direction)

        # Construct an initial "look-at" rotation from direction
        up = np.array([0, 1, 0], dtype=float)
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.vstack([right, up, direction]).T  # shape (3, 3)

        # Create small rotations around X/Y axes from the angles
        rotation_x = transform.from_rotvec(vertical_angle * np.array([1, 0, 0], dtype=float))
        rotation_y = transform.from_rotvec(-horizontal_angle * np.array([0, 1, 0], dtype=float))

        # Combine: (apply y-rotation), then the base look-at rotation, then x-rotation
        base_rotation = transform.from_matrix(rotation_matrix)
        rotation_combined = rotation_y * base_rotation * rotation_x

        # Convert the final rotation to Euler angles (XYZ order, in degrees)
        euler_angles = rotation_combined.as_euler('xyz', degrees=True)

        # Append a constant focal length (or any other parameter) as 7th DoF
        focal_length = 955.02
        camera_tensor = np.hstack([translation, euler_angles, [focal_length]])

        camera_data.append(camera_tensor)

    # Return the final camera data as a torch tensor
    return torch.tensor(camera_data, dtype=torch.float)


def test_ccd_transform_to_7DoF():
    """
    A simple test to ensure that ccd_transform_to_7DoF runs without error and returns
    the correct output shape for a given input.
    """

    # Create a dummy trajectory of shape (1, N, 5).
    # For instance, we have N=5 frames, each with:
    #   [x, y, z, horizontal_factor, vertical_factor]
    dummy_traj = np.array([[
        [1.0,  2.0,  3.0,  0.1, 0.2],
        [4.0,  5.0,  6.0,  0.2, 0.3],
        [7.0,  8.0,  9.0,  0.3, 0.4],
        [10.0, 11.0, 12.0, 0.4, 0.5],
        [13.0, 14.0, 15.0, 0.5, 0.6],
    ]])

    # Call the function
    output = ccd_transform_to_7DoF(dummy_traj)

    print("Output shape:", output.shape)
    print("Output tensor:\n", output)
    # We expect shape = [30, 7], since by default it downsamples/upsamples to 30 frames
    #  (x, y, z, euler_x, euler_y, euler_z, focal_length)

# Uncomment to run the test:
# test_ccd_transform_to_7DoF()
