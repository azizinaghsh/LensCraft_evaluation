import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

from scipy.spatial.transform import Slerp

def resize_trajectory(trajectories, target_frames=30):
    """
    Upsample or downsample the trajectories to have the specified number of frames.
    :param trajectories: Input array of trajectories of shape [N, T, 4, 4] where N is the number of trajectories
                         and T is the number of frames.
    :param target_frames: Desired number of frames for each trajectory.
    :return: Resized trajectories with shape [N, target_frames, 4, 4].
    """
    N, T, _, _ = trajectories.shape
    resized_trajectories = []

    for i in range(N):
        # Resample the trajectory to the target number of frames
        times = np.linspace(0, 1, T)  # Original times
        target_times = np.linspace(0, 1, target_frames)  # Target times
        
        positions = trajectories[i][:, :3, 3]  # Extract position components
        interpolator = interp1d(times, positions, axis=0, kind='linear', fill_value='extrapolate')
        resized_positions = interpolator(target_times)  # Interpolated positions


        # Interpolate rotations using Slerp
        rotation_matrices = trajectories[i][:, :3, :3]
        rotations = R.from_matrix(rotation_matrices)
        slerp = Slerp(times, rotations)
        resized_rotations = slerp(target_times).as_matrix()

        # Combine positions and rotations into a resized trajectory
        resized_trajectory = np.zeros((target_frames, 4, 4))
        resized_trajectory[:, :3, :3] = resized_rotations
        resized_trajectory[:, :3, 3] = resized_positions
        resized_trajectory[:, 3, 3] = 1  # Homogeneous coordinate
        
        resized_trajectories.append(resized_trajectory)
    
    return np.array(resized_trajectories)



def trajectory_to_7dof(trajectories):
    """
    Convert a batch of camera trajectories to 7DoF representation: 
    3 positions (x, y, z), 3 Euler angles (pitch, yaw, roll), and 1 FoV.
    :param trajectories: Input array of camera trajectories of shape [N, T, 4, 4].
    :return: Converted trajectories of shape [N, T, 7], where each trajectory consists of
             3 positions, 3 Euler angles, and 1 fixed FoV (60 degrees).
    """
    N, T, _, _ = trajectories.shape
    result = np.zeros((N, T, 7))  # Output with shape [N, T, 7]


    for i in range(N):
        for t in range(T):
            # Extract position (translation vector)
            position = trajectories[i][t, :3, 3]
            # Convert rotation matrix to Euler angles (pitch, yaw, roll)
            rotation_matrix = trajectories[i][t, :3, :3]
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler("xyz", degrees=True)
            
            # Set position, Euler angles, and FoV (60 degrees)
            result[i, t, :3] = position
            result[i, t, 3:6] = euler_angles
            result[i, t, 6] = 60  # Set fixed FoV to 60 degrees
    
    return result

def transform_trajectories(trajectories, target_frames=30):
    """
    Apply the transformations to a batch of camera trajectories:
    1. Resize to the target number of frames (e.g., 30).
    2. Convert each frame to 7DoF (3 positions, 3 Euler angles, 1 fixed FoV).
    :param trajectories: Input array of camera trajectories of shape [N, T, 4, 4].
    :param target_frames: Desired number of frames for the output.
    :return: Transformed camera trajectories of shape [N, 30, 7].
    """
    resized_trajectories = resize_trajectory(trajectories, target_frames)
    pos = resized_trajectories[0, :, :3, 3]
    t = trajectory_to_7dof(resized_trajectories)
    
    return t


def generate_valid_transformation_matrix():
    # Generate a random rotation matrix (3x3) using scipy.spatial.transform
    rotation = R.random()  # Random rotation
    rotation_matrix = rotation.as_matrix()

    # Generate a random translation vector (3,)
    translation = np.random.rand(3)

    # Construct the 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Start with identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Assign rotation part
    transformation_matrix[:3, 3] = translation  # Assign translation part

    return transformation_matrix

def test_transform_trajectories():
    # Test the transformation process with valid input
    N = 2  # Number of trajectories
    T = 10  # Number of frames per trajectory

    # Generate a batch of valid transformation matrices (N trajectories, T frames)
    trajectories = np.array([[
        generate_valid_transformation_matrix() for _ in range(T)
    ] for _ in range(N)])

    # Transform trajectories to 30 frames and 7DoF representation
    transformed_trajectories = transform_trajectories(trajectories, target_frames=30)

    # Check the shape of the result
    assert transformed_trajectories.shape == (N, 30, 7), \
        f"Expected shape (N, 30, 7), got {transformed_trajectories.shape}"

    # Check that the transformed data is correctly formatted (positions, Euler angles, and FoV)
    for i in range(N):
        for t in range(30):
            pos = transformed_trajectories[i, t, :3]
            euler_angles = transformed_trajectories[i, t, 3:6]
            fov = transformed_trajectories[i, t, 6]

            assert pos.shape == (3,), f"Expected position shape (3,), got {pos.shape}"
            assert euler_angles.shape == (3,), f"Expected Euler angles shape (3,), got {euler_angles.shape}"
            assert np.isclose(fov, 60), f"Expected FoV 60, got {fov}"

    print("Test passed successfully.")

# Run the test
test_transform_trajectories()
