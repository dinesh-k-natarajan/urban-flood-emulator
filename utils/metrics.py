import torch
import numpy as np
from torchmetrics import Metric

class PixelwiseCoefficientOfDetermination(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_r2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        N, C, H, W = preds.shape
        # Flatten the tensors to shape [N, C * H * W]
        preds_flat = preds.view(N, -1)
        target_flat = target.view(N, -1)
        # Compute the mean for each pixel position across batches
        target_mean = torch.mean(target_flat, dim=-1, keepdim=True)
        # Residual Sum of Squares (SSR)
        ssr = torch.sum((target_flat - preds_flat) ** 2, dim=-1)
        # Total Sum of Squares (SST)
        sst = torch.sum((target_flat - target_mean) ** 2, dim=-1)
        # Compute R² for each pixel
        r2 = 1 - ssr / (sst + 1e-8)
        # Average R² across all pixels
        mean_r2 = torch.mean(r2)
        # Update the states
        self.sum_r2 += mean_r2 * N  # Sum up R² values
        self.total += N  # Count the number of images

    def compute(self):
        # Compute the final average R²
        return self.sum_r2 / self.total
