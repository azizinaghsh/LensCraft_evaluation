from omegaconf.dictconfig import DictConfig
from typing import List, Tuple

from ema_pytorch import EMA
import numpy as np
import torch
from torchtyping import TensorType
import torch.nn as nn
import lightning as L

from utils.random_utils import StackedRandomGenerator

# ------------------------------------------------------------------------------------- #

batch_size, num_samples = None, None
num_feats, num_rawfeats, num_cams = None, None, None
RawTrajectory = TensorType["num_samples", "num_rawfeats", "num_cams"]

# ------------------------------------------------------------------------------------- #


class Diffuser(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        guidance_weight: float,
        ema_kwargs: DictConfig,
        sampling_kwargs: DictConfig,
        edm2_normalization: bool,
        **kwargs,
    ):
        super().__init__()

        # Network and EMA
        self.net = network
        self.ema = EMA(self.net, **ema_kwargs)
        self.guidance_weight = guidance_weight
        self.edm2_normalization = edm2_normalization
        self.sigma_data = network.sigma_data

        # Sampling
        self.num_steps = sampling_kwargs.num_steps
        self.sigma_min = sampling_kwargs.sigma_min
        self.sigma_max = sampling_kwargs.sigma_max
        self.rho = sampling_kwargs.rho
        self.S_churn = sampling_kwargs.S_churn
        self.S_noise = sampling_kwargs.S_noise
        self.S_min = sampling_kwargs.S_min
        self.S_max = (
            sampling_kwargs.S_max
            if isinstance(sampling_kwargs.S_max, float)
            else float("inf")
        )

    # ---------------------------------------------------------------------------------- #

    def on_predict_start(self):
        eval_dataset = self.trainer.datamodule.eval_dataset
        self.modalities = list(eval_dataset.modality_datasets.keys())

        self.get_matrix = self.trainer.datamodule.train_dataset.get_matrix
        self.v_get_matrix = self.trainer.datamodule.eval_dataset.get_matrix

    def predict_step(self, batch, batch_idx):
        ref_samples, mask = batch["traj_feat"], batch["padding_mask"]

        if len(self.modalities) > 0:
            cond_k = [x for x in batch.keys() if "traj" not in x and "feat" in x]
            cond_data = [batch[cond] for cond in cond_k]
            conds = {}
            for cond in cond_k:
                cond_name = cond.replace("_feat", "")
                if isinstance(batch[f"{cond_name}_raw"], dict):
                    for cond_name_, x in batch[f"{cond_name}_raw"].items():
                        conds[cond_name_] = x
                else:
                    conds[cond_name] = batch[f"{cond_name}_raw"]
            batch["conds"] = conds
        else:
            cond_data = None

        # cf edm2 sigma_data normalization / https://arxiv.org/pdf/2312.02696.pdf
        if self.edm2_normalization:
            ref_samples *= self.sigma_data
        _, gen_samples = self.sample(self.ema.ema_model, ref_samples, cond_data, mask)

        batch["ref_samples"] = torch.stack([self.v_get_matrix(x) for x in ref_samples])
        batch["gen_samples"] = torch.stack([self.get_matrix(x) for x in gen_samples])

        return batch

    # --------------------------------------------------------------------------------- #

    def sample(
        self,
        net: torch.nn.Module,
        traj_samples: RawTrajectory,
        cond_samples: TensorType["num_samples", "num_feats"],
        mask: TensorType["num_samples", "num_feats"],
        external_seeds: List[int] = None,
    ) -> Tuple[RawTrajectory, RawTrajectory]:
        # Pick latents
        num_samples = traj_samples.shape[0]
        seeds = self.gen_seeds if hasattr(self, "gen_seeds") else range(num_samples)
        rnd = StackedRandomGenerator(self.device, seeds)

        sz = [num_samples, self.net.num_feats, self.net.num_cams]
        latents = rnd.randn_rn(sz, device=self.device)
        # Generate trajectories.
        generations = self.edm_sampler(
            net,
            latents,
            class_labels=cond_samples,
            mask=mask,
            randn_like=rnd.randn_like,
            guidance_weight=self.guidance_weight,
            # ----------------------------------- #
            num_steps=self.num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            S_churn=self.S_churn,
            S_min=self.S_min,
            S_max=self.S_max,
            S_noise=self.S_noise,
        )

        return latents, generations

    @staticmethod
    def edm_sampler(
        net,
        latents,
        class_labels=None,
        mask=None,
        guidance_weight=2.0,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        # Time step discretization.
        step_indices = torch.arange(num_steps, device=latents.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        bool_mask = ~mask.to(bool)
        x_next = latents * t_steps[0]
        bs = latents.shape[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            if class_labels is not None:
                class_label_knot = [torch.zeros_like(label) for label in class_labels]
                x_hat_both = torch.cat([x_hat, x_hat], dim=0)
                y_label_both = [
                    torch.cat([y, y_knot], dim=0)
                    for y, y_knot in zip(class_labels, class_label_knot)
                ]

                bool_mask_both = torch.cat([bool_mask, bool_mask], dim=0)
                t_hat_both = torch.cat([t_hat.expand(bs), t_hat.expand(bs)], dim=0)
                cond_denoised, denoised = net(
                    x_hat_both, t_hat_both, y=y_label_both, mask=bool_mask_both
                ).chunk(2, dim=0)
                denoised = denoised + (cond_denoised - denoised) * guidance_weight
            else:
                denoised = net(x_hat, t_hat.expand(bs), mask=bool_mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                if class_labels is not None:
                    class_label_knot = [
                        torch.zeros_like(label) for label in class_labels
                    ]
                    x_next_both = torch.cat([x_next, x_next], dim=0)
                    y_label_both = [
                        torch.cat([y, y_knot], dim=0)
                        for y, y_knot in zip(class_labels, class_label_knot)
                    ]
                    bool_mask_both = torch.cat([bool_mask, bool_mask], dim=0)
                    t_next_both = torch.cat(
                        [t_next.expand(bs), t_next.expand(bs)], dim=0
                    )
                    cond_denoised, denoised = net(
                        x_next_both, t_next_both, y=y_label_both, mask=bool_mask_both
                    ).chunk(2, dim=0)
                    denoised = denoised + (cond_denoised - denoised) * guidance_weight
                else:
                    denoised = net(x_next, t_next.expand(bs), mask=bool_mask)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
