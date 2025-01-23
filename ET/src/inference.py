from typing import Any, Callable, Dict
import torch

from ET.utils.common_viz import init, get_batch
from ET.utils.random_utils import set_random_seed
from ET.utils.rerun import et_log_sample
from ET.src.diffuser import Diffuser
from ET.src.datasets.multimodal_dataset import MultimodalDataset
from torch.utils.data import Dataset


def generate(
    prompt: str,
    seed: int,
    guidance_weight: float,
    character_position: list,
    # ----------------------- #
    dataset: Dataset,
    device: torch.device,
    clip_model: clip.model.CLIP,
) -> Dict[str, Any]:
    diffuser.to(device)
    clip_model.to(device)

    # Set arguments
    set_random_seed(seed)
    diffuser.gen_seeds = np.array([seed])
    diffuser.guidance_weight = guidance_weight

    # Inference
    sample_id = SAMPLE_IDS[0]  # Default to the first sample ID
    seq_feat = diffuser.net.model.clip_sequential

    batch = get_batch(prompt, sample_id, character_position, clip_model, dataset, seq_feat, device)

    with torch.no_grad():
        out = diffuser.predict_step(batch, 0)

    # Run visualization
    padding_mask = out["padding_mask"][0].to(bool).cpu()
    padded_traj = out["gen_samples"][0].cpu()
    traj = padded_traj[padding_mask]
    char_traj = out["char_feat"][0].cpu()
    padded_vertices = out["char_raw"]["char_vertices"][0]
    vertices = padded_vertices[padding_mask]
    faces = out["char_raw"]["char_faces"][0]
    normals = get_normals(vertices, faces)
    fx, fy, cx, cy = out["intrinsics"][0].cpu().numpy()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    caption = out["caption_raw"][0]