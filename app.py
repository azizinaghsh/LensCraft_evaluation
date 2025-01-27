import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
et_dir = os.path.join(current_dir, 'ET')
ccd_dir = os.path.join(current_dir, 'CCD')
sys.path.append(et_dir)
sys.path.append(ccd_dir)

from functools import partial
from typing import Any, Callable, Dict, List

import clip
import numpy as np
import torch

    from ET.utils.common_viz import init, get_batch
    from ET.utils.random_utils import set_random_seed
    from ET.src.diffuser import Diffuser
    from ET.src.datasets.multimodal_dataset import MultimodalDataset
    from ET.utils.transform import resize_trajectory, trajectory_to_7dof

def generate_batch(
    prompts: List[str],
    seed: int,
    guidance_weight: float,
    character_position: list,
    # ----------------------- #
    dataset: MultimodalDataset,
    device: torch.device,
    diffuser: Diffuser,
    clip_model: clip.model.CLIP,
    target_frames: int = 30  # New parameter for target number of frames
) -> List[Dict[str, Any]]:
    """
    Generate samples for a batch of prompts concurrently on the GPU,
    and return transformed trajectories in [N, 30, 7] format.
    """
    diffuser.to(device)
    clip_model.to(device)

    # Set arguments
    set_random_seed(seed)
    diffuser.gen_seeds = np.array([seed])
    diffuser.guidance_weight = guidance_weight

    # Generate batch of data
    batch = get_batch(prompts, character_position, clip_model, dataset, device)

    # Inference for the whole batch
    with torch.no_grad():
        out = diffuser.predict_step(batch, 0)

    # Initialize result list for the transformed data
    results = []

    # Process each sample in the batch
    for i in range(len(prompts)):
        # Extract trajectory and other relevant information
        padding_mask = out["padding_mask"][i].to(bool).cpu()
        padded_traj = out["gen_samples"][i].cpu()
        traj = padded_traj[padding_mask]
        char_traj = out["char_feat"][i].cpu()

        # Resize trajectory to have 30 frames
        resized_traj = resize_trajectory(traj, target_frames)

        # Convert trajectory to 7DoF format (3 positions, 3 Euler angles, 1 FoV)
        traj_7dof = trajectory_to_7dof(resized_traj)

        results.append({"traj_7dof": traj_7dof, "char_traj": char_traj})

    return results
