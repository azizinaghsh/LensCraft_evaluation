from typing import Any, Dict, List

from metrics.modules.fcd import FrechetCLaTrDistance
from metrics.modules.prdc import ManifoldMetrics
from metrics.modules.clatr_score import CLaTrScore


class MetricCallback:
    def __init__(
        self,
        num_cams: int,
        device: str,
    ):
        self.num_cams = num_cams


        self.clatr_fd = {
            "train": FrechetCLaTrDistance(),
            "val": FrechetCLaTrDistance(),
            "test": FrechetCLaTrDistance(),
        }
        self.clatr_prdc = {
            "train": ManifoldMetrics(distance="euclidean"),
            "val": ManifoldMetrics(distance="euclidean"),
            "test": ManifoldMetrics(distance="euclidean"),
        }
        self.clatr_score = {
            "train": CLaTrScore(),
            "val": CLaTrScore(),
            "test": CLaTrScore(),
        }

        self.device = device
        self._move_to_device(device)

    def _move_to_device(self, device: str):
        for stage in ["train", "val", "test"]:
            self.clatr_fd[stage].to(device)
            self.clatr_prdc[stage].to(device)
            self.clatr_score[stage].to(device)


    def update_clatr_metrics(self, stage, pred, ref, text):
        self.clatr_score[stage].update(pred, text)
        self.clatr_prdc[stage].update(pred, ref)
        self.clatr_fd[stage].update(pred, ref)

    def compute_clatr_metrics(self, stage: str) -> Dict[str, Any]:
        clatr_score = self.clatr_score[stage].compute()
        self.clatr_score[stage].reset()

        clatr_p, clatr_r, clatr_d, clatr_c = self.clatr_prdc[stage].compute()
        self.clatr_prdc[stage].reset()

        fcd = self.clatr_fd[stage].compute()
        self.clatr_fd[stage].reset()

        return {
            "clatr/clatr_score": clatr_score,
            "clatr/precision": clatr_p,
            "clatr/recall": clatr_r,
            "clatr/density": clatr_d,
            "clatr/coverage": clatr_c,
            "clatr/fcd": fcd,
        }
