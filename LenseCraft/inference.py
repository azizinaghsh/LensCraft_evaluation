import torch
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from data.simulation.dataset import SimulationDataset
from data.datamodule import CameraTrajectoryDataModule

@dataclass
class TrajectoryData:
    subject_trajectory: torch.Tensor
    camera_trajectory: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    src_key_mask: Optional[torch.Tensor] = None
    caption_feat: Optional[torch.Tensor] = None
    teacher_forcing_ratio: Optional[int] = 0.0


def infer(dataset : SimulationDataset, cfg, device):
    GlobalHydra.instance().clear()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
        
    model = _initialize_model(cfg)

    # device = torch.device(
    #         cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")

    return process_samples(model, dataset, device)

def _initialize_model(self, cfg: DictConfig) -> torch.nn.Module:
    model = instantiate(cfg.training.model)
    model = load_checkpoint(cfg.checkpoint_path, model, self.device)
    model.to(self.device)
    model.eval()
    return model    

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model

def setup_data_module(cfg: DictConfig) -> CameraTrajectoryDataModule:
    data_module = CameraTrajectoryDataModule(
        dataset_config=cfg.data.dataset.module,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size
    )
    data_module.setup()
    return data_module

def get_sample_indices(cfg: DictConfig, dataset) -> list:
    if cfg.sample_id:
        for idx in range(len(dataset)):
            if dataset.original_dataset.root_filenames[idx] == cfg.sample_id:
                return [idx]
        raise ValueError(f"Sample ID {cfg.sample_id} not found in dataset")
    return list(range(10))

def process_samples(
    model,
    dataset,
    device,
):
    sample_indices = list(range(len(dataset)))

    generated_samples = []

    for idx in sample_indices:
        sample = dataset[idx]
        data = TrajectoryData(
            subject_trajectory=sample['subject_trajectory'].unsqueeze(0),
            camera_trajectory=sample['camera_trajectory'].unsqueeze(0),
            padding_mask=sample.get('padding_mask', None),
            caption_feat=torch.stack([
                sample['movement_clip'],
                sample['easing_clip'],
                sample['angle_clip'],
                sample['shot_clip']
            ]).unsqueeze(1)
        )
        data.teacher_forcing_ratio = 1.0
        prompt_gen = reconstruct_trajectory(model, data, device)
        generated_samples.append(prompt_gen)

    return torch.stack(generated_samples)


def reconstruct_trajectory(
    model,
    data: TrajectoryData,
    device
) -> Optional[torch.Tensor]:
    with torch.no_grad():
        padding_mask = data.padding_mask.to(device) if data.padding_mask is not None else None
        src_key_mask = data.src_key_mask.to(device) if data.src_key_mask is not None else None
        caption_feat = data.caption_feat.to(device)
        
        output = model(
            data.camera_trajectory.to(device),
            data.subject_trajectory.to(device),
            clip_embeddings=caption_feat,
            teacher_forcing_ratio=data.teacher_forcing_ratio,
            src_key_mask=src_key_mask,
            tgt_key_padding_mask=padding_mask
        )['reconstructed'].squeeze(0)
        
        return output