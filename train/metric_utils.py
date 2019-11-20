import torch

def snr(pred_signal: torch.Tensor, true_signal: torch.Tensor) -> torch.FloatTensor:
    """
        Calculate the Signal-to-Noise Ratio
        from two signals
        
        arguments:
            pred_signal: predicted signal spectrogram, expected shape: (N, 2, width, height)
            true_signal: original signal spectrogram, expected shape: (N, 2, width, height)
    """
    inter_signal = true_signal - pred_signal
    
    true_power = (true_signal ** 2).sum()
    inter_power = (inter_signal ** 2).sum()
    
    snr = 10*torch.log10(true_power / inter_power)
    
    return snr
