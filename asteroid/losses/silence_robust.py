import torch
from .sdr import PairwiseNegSDR

class SilenceRobustPairwiseNegSDR(PairwiseNegSDR):
    """
    Pairwise Negative SI-SDR that handles silent targets by minimizing the power of the estimate.
    
    If the target energy is below a threshold, the loss becomes 10 * log10(||est||^2 + EPS).
    This ensures that the model learns to output silence for silent targets.
    """
    def __init__(self, *args, threshold=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        
        # Calculate energy of targets to identify silence
        # [batch, n_src]
        target_energy = torch.sum(targets**2, dim=-1)
        
        # Compute standard PairwiseNegSDR
        # This might result in NaNs/Infs for silent targets, but we will mask them out.
        # We use the parent implementation but we need to capture the result.
        # Since we cannot easily "patch" the parent method's internal vars, 
        # we basically have to reimplement the logic or call super and fix it.
        # Calling super might crash if it does division by zero energy without checks.
        # The base implementation adds EPS to energy: 
        # s_target_energy = torch.sum(s_target**2, dim=3, keepdim=True) + self.EPS
        # So it shouldn't crash, but it will produce garbage (large values).
        
        pairwise_loss = super().forward(est_targets, targets)
        # [batch, n_src, n_src] (n_est, n_target)
        # Note: PairwiseNegSDR returns shape (batch, n_est, n_target) if I recall correctly from reading code?
        # Let's double check code in sdr.py. 
        # It computes pair_wise_dot [batch, n_src, n_src, 1].
        # It assumes est_targets and targets have same n_src.
        # Returns [batch, n_src, n_src].
        # Rows are estimates, Columns are targets (or vice versa depending on broadcasting).
        # s_target is unsqueeze(dim=1) -> [batch, 1, n_src, time]
        # s_estimate is unsqueeze(dim=2) -> [batch, n_src, 1, time]
        # pair_wise_dot = sum(s_estimate * s_target) -> sum over time.
        # So index (i, j) corresponds to est[i] and target[j].
        
        # Calculate silence loss for each estimate
        # [batch, n_src]
        est_energy = torch.sum(est_targets**2, dim=-1)
        # [batch, n_src]
        silence_loss_vec = 10 * torch.log10(est_energy + self.EPS)
        
        # We need to broadcast silence_loss_vec to the pairwise matrix.
        # The pairwise matrix is (batch, n_est, n_target).
        # If target j is silent, then for all estimates i, the loss(i, j) should be silence_loss_vec(i).
        # Wait. 
        # If target j is silence. We want estimate i to match it?
        # PIT will try to assign estimate i to target j.
        # If it assigns i -> j, we want estimate i to be silence.
        # So yes, loss(i, j) should be silence_loss(estimate i).
        
        # Expand silence_loss_vec to [batch, n_est, 1]
        silence_loss_matrix = silence_loss_vec.unsqueeze(2).expand_as(pairwise_loss)
        
        # Mask for silent targets
        # target_energy is [batch, n_src] (targets)
        # We need [batch, 1, n_src] to match (batch, est, target)
        is_silent = (target_energy < self.threshold).unsqueeze(1)
        
        # Combine
        # If silent, use silence_loss. Else use pairwise_loss.
        final_loss = torch.where(is_silent, silence_loss_matrix, pairwise_loss)
        
        return final_loss
