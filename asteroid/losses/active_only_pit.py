import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.modules.loss import _Loss

from .sdr import PairwiseNegSDR


class ActiveOnlyPITSilencePenalty(_Loss):
    """PIT loss on active targets only, plus bounded silence penalty on extras.

    This criterion is designed for fixed-output models with variable numbers of
    active speakers. Active targets drive the permutation-based SI-SDR objective.
    Unused estimate channels are constrained toward silence with a margin penalty.
    """

    def __init__(
        self,
        threshold=1e-5,
        silence_margin_db=-45.0,
        silence_metric="rms_db",
        active_weight=1.0,
        silence_weight=0.2,
        active_metric="sisdr",
        active_threshold_mode="rms",
        active_l1_weight=0.0,
        active_l2_weight=0.0,
        active_rms_floor_db=None,
        active_rms_penalty_weight=0.0,
        active_diversity_weight=0.0,
        active_activity_bce_weight=0.0,
        activity_threshold_db=-40.0,
        activity_temp_db=3.0,
        eps=1e-8,
    ):
        super().__init__()
        self.threshold = threshold
        self.silence_margin_db = silence_margin_db
        self.silence_metric = silence_metric
        self.active_weight = active_weight
        self.silence_weight = silence_weight
        self.active_metric = active_metric
        self.active_threshold_mode = active_threshold_mode
        self.active_l1_weight = active_l1_weight
        self.active_l2_weight = active_l2_weight
        self.active_rms_floor_db = active_rms_floor_db
        self.active_rms_penalty_weight = active_rms_penalty_weight
        self.active_diversity_weight = active_diversity_weight
        self.active_activity_bce_weight = active_activity_bce_weight
        self.activity_threshold_db = activity_threshold_db
        self.activity_temp_db = activity_temp_db
        self.eps = eps
        if self.active_metric not in ("sisdr", "sdsdr"):
            raise ValueError(
                f"Unsupported active_metric={self.active_metric}. Use 'sisdr' or 'sdsdr'."
            )
        if self.silence_metric not in ("rms_db", "energy_db"):
            raise ValueError(
                f"Unsupported silence_metric={self.silence_metric}. Use 'rms_db' or 'energy_db'."
            )
        if self.active_threshold_mode not in ("rms", "sum_energy"):
            raise ValueError(
                "Unsupported active_threshold_mode="
                f"{self.active_threshold_mode}. Use 'rms' or 'sum_energy'."
            )
        self.pairwise_sisdr = PairwiseNegSDR("sisdr")
        self.pairwise_sdsdr = PairwiseNegSDR("sdsdr")

    def _silence_penalty(self, est_targets):
        if self.silence_metric == "rms_db":
            est_rms = torch.sqrt(torch.mean(est_targets**2, dim=-1) + self.eps)
            est_db = 20 * torch.log10(est_rms + self.eps)
        else:
            est_energy = torch.sum(est_targets**2, dim=-1) + self.eps
            est_db = 10 * torch.log10(est_energy)
        # Zero penalty once channel is quieter than configured margin.
        return torch.relu(est_db - self.silence_margin_db)

    def _target_activity(self, targets):
        if self.active_threshold_mode == "rms":
            return torch.mean(targets**2, dim=-1)
        return torch.sum(targets**2, dim=-1)

    def _channel_rms_db(self, est_targets):
        est_rms = torch.sqrt(torch.mean(est_targets**2, dim=-1) + self.eps)
        return 20 * torch.log10(est_rms + self.eps)

    def _active_diversity_penalty(self, est_active):
        if est_active.ndim != 2 or est_active.size(0) < 2:
            return est_active.new_tensor(0.0)
        centered = est_active - est_active.mean(dim=-1, keepdim=True)
        norms = torch.sqrt(torch.sum(centered**2, dim=-1, keepdim=True) + self.eps)
        normalized = centered / norms
        corr = torch.matmul(normalized, normalized.transpose(0, 1))
        off_diag = corr - torch.eye(corr.size(0), device=corr.device, dtype=corr.dtype)
        return torch.mean(off_diag**2)

    def forward(self, est_targets, targets, return_components=False):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                "Inputs must be of shape [batch, n_src, time], got "
                f"{targets.size()} and {est_targets.size()} instead"
            )

        batch_size, n_src, _ = targets.shape
        pairwise_sisdr = self.pairwise_sisdr(est_targets, targets)
        pairwise_active = pairwise_sisdr
        if self.active_metric == "sdsdr":
            pairwise_active = self.pairwise_sdsdr(est_targets, targets)
        target_activity = self._target_activity(targets)
        silence_penalty = self._silence_penalty(est_targets)

        losses = []
        active_pair_losses = []
        active_l1_losses = []
        active_l2_losses = []
        active_rms_floor_losses = []
        active_diversity_losses = []
        active_activity_bce_losses = []
        silent_losses = []
        n_active_targets = []
        for b in range(batch_size):
            active_tgt_idx = torch.where(target_activity[b] > self.threshold)[0]
            n_active_targets.append(active_tgt_idx.numel())
            if active_tgt_idx.numel() > 0:
                # Rectangular assignment: estimates x active targets.
                cost = pairwise_active[b, :, active_tgt_idx].detach().cpu().numpy()
                est_rows, tgt_cols_local = linear_sum_assignment(cost)
                est_rows_t = torch.as_tensor(est_rows, device=est_targets.device, dtype=torch.long)
                tgt_cols_t = active_tgt_idx[
                    torch.as_tensor(tgt_cols_local, device=est_targets.device, dtype=torch.long)
                ]
                active_pair_loss = pairwise_active[b, est_rows_t, tgt_cols_t].mean()

                # Scale-sensitive losses on matched active channels to discourage
                # low-amplitude collapse that SI-SDR alone can permit.
                est_active = est_targets[b, est_rows_t]
                tgt_active = targets[b, tgt_cols_t]
                active_l1 = torch.mean(torch.abs(est_active - tgt_active))
                active_l2 = torch.mean((est_active - tgt_active) ** 2)
                if self.active_rms_floor_db is not None and self.active_rms_penalty_weight > 0.0:
                    est_rms = torch.sqrt(torch.mean(est_active**2, dim=-1) + self.eps)
                    est_rms_db = 20 * torch.log10(est_rms + self.eps)
                    active_rms_penalty = torch.relu(self.active_rms_floor_db - est_rms_db).mean()
                else:
                    active_rms_penalty = pairwise_sisdr.new_tensor(0.0)
                active_diversity = self._active_diversity_penalty(est_active)
            else:
                est_rows_t = torch.empty(0, dtype=torch.long, device=est_targets.device)
                active_pair_loss = pairwise_sisdr.new_tensor(0.0)
                active_l1 = pairwise_sisdr.new_tensor(0.0)
                active_l2 = pairwise_sisdr.new_tensor(0.0)
                active_rms_penalty = pairwise_sisdr.new_tensor(0.0)
                active_diversity = pairwise_sisdr.new_tensor(0.0)

            est_rms_db_all = self._channel_rms_db(est_targets[b])
            activity_logits = (est_rms_db_all - self.activity_threshold_db) / max(
                self.activity_temp_db, self.eps
            )
            activity_probs = torch.sigmoid(activity_logits)
            activity_target = torch.zeros_like(activity_probs)
            if est_rows_t.numel() > 0:
                activity_target[est_rows_t] = 1.0
            active_activity_bce = torch.nn.functional.binary_cross_entropy(
                activity_probs, activity_target
            )

            active_loss = (
                active_pair_loss
                + self.active_l1_weight * active_l1
                + self.active_l2_weight * active_l2
                + self.active_rms_penalty_weight * active_rms_penalty
                + self.active_diversity_weight * active_diversity
                + self.active_activity_bce_weight * active_activity_bce
            )

            if est_rows_t.numel() < n_src:
                mask = torch.ones(n_src, dtype=torch.bool, device=est_targets.device)
                mask[est_rows_t] = False
                silent_loss = silence_penalty[b, mask].mean()
            else:
                silent_loss = pairwise_sisdr.new_tensor(0.0)

            total = self.active_weight * active_loss + self.silence_weight * silent_loss
            losses.append(total)
            active_pair_losses.append(active_pair_loss)
            active_l1_losses.append(active_l1)
            active_l2_losses.append(active_l2)
            active_rms_floor_losses.append(active_rms_penalty)
            active_diversity_losses.append(active_diversity)
            active_activity_bce_losses.append(active_activity_bce)
            silent_losses.append(silent_loss)

        mean_total = torch.stack(losses, dim=0).mean()
        if not return_components:
            return mean_total

        components = {
            "active_pair": torch.stack(active_pair_losses, dim=0).mean(),
            "active_l1": torch.stack(active_l1_losses, dim=0).mean(),
            "active_l2": torch.stack(active_l2_losses, dim=0).mean(),
            "active_rms_floor": torch.stack(active_rms_floor_losses, dim=0).mean(),
            "active_diversity": torch.stack(active_diversity_losses, dim=0).mean(),
            "active_activity_bce": torch.stack(active_activity_bce_losses, dim=0).mean(),
            "silent_rms_penalty": torch.stack(silent_losses, dim=0).mean(),
            "n_active_targets": torch.tensor(
                float(sum(n_active_targets)) / max(len(n_active_targets), 1),
                device=est_targets.device,
            ),
        }
        return mean_total, components
