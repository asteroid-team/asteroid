from metric_utils import snr
from catalyst.dl.core import MetricCallback


class SNRCallback(MetricCallback):
    """
    SNR callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "snr"
    ):
        """
        Args:
            input_key (str): input key to use for dice calculation;
                specifies our y_true.
            output_key (str): output key to use for dice calculation;
                specifies our y_pred.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=snr,
            input_key=input_key,
            output_key=output_key
        )
    
    def on_batch_end(self, state):
        output_audios = state.output[self.output_key]
        true_audios = state.input[self.input_key]
        
        num_person = state.model.num_person
        
        avg_snr = 0
        for n in range(num_person):
            output_audio = output_audios[..., n]
            true_audio = true_audios[..., n]
            
            snr_value = snr(output_audio, true_audio).item()
            avg_snr += (snr_value - avg_snr) / num_person
        
        state.metrics.add_batch_value(name=self.prefix, value=avg_snr)
