import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
et_dir = os.path.join(current_dir, 'ET')
ccd_dir = os.path.join(current_dir, 'CCD')
lens_dir = os.path.join(current_dir, 'LensCraft')
sys.path.append(et_dir)
sys.path.append(ccd_dir)
sys.path.append(lens_dir)

from functools import partial
from typing import Any, Callable, Dict

import clip
import numpy as np
import torch

from ET.utils.common_viz import init, get_batch
from ET.utils.random_utils import set_random_seed
from ET.utils.rerun import et_log_sample
from ET.src.diffuser import Diffuser
from ET.src.datasets.multimodal_dataset import MultimodalDataset

from CCD.utils.transform import ccd_transform_to_7DoF
from CCD.src.main import generate_CCD_sample



def generate_ccd(
    prompt: str,
    seed: int,
    guidance_weight: float,
    character_position: list,
) -> Dict[str, Any]:
    
    results = generate_CCD_sample(prompt, seed)

    ccd_transform_to_7DoF(
        traj=np.array(results)
    )
    return "./.tmp_gr.rrd"

def generate(
    prompt: str,
    seed: int,
    guidance_weight: float,
    character_position: list,
    # ----------------------- #
    dataset: MultimodalDataset,
    device: torch.device,
    diffuser: Diffuser,
    clip_model: clip.model.CLIP,
) -> Dict[str, Any]:
    diffuser.to(device)
    clip_model.to(device)

    # Set arguments
    set_random_seed(seed)
    diffuser.gen_seeds = np.array([seed])
    diffuser.guidance_weight = guidance_weight

    # Inference
    sample_id = 0
    seq_feat = diffuser.net.model.clip_sequential

    batch = get_batch(prompt, sample_id, character_position, clip_model, dataset, seq_feat, device)

    with torch.no_grad():
        out = diffuser.predict_step(batch, 0)

    padding_mask = out["padding_mask"][0].to(bool).cpu()
    padded_traj = out["gen_samples"][0].cpu()
    traj = padded_traj[padding_mask]
    char_traj = out["char_feat"][0].cpu()
    fx, fy, cx, cy = out["intrinsics"][0].cpu().numpy()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    et_log_sample(
        root_name="world",
        traj=traj.numpy(),
        char_traj=char_traj.numpy(),
        K=K,
    )



# ------------------------------------------------------------------------------------- #

diffuser, clip_model, dataset, device = init("config")
generate_sample_et = partial(
    generate,
    dataset=dataset,
    device=device,
    diffuser=diffuser,
    clip_model=clip_model,
)

generate_sample_ccd = partial(
    generate_ccd
)
