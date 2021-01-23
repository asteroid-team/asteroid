from torch import nn
from .base_models import BaseEncoderMaskerDecoder
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, magreim
from ..masknn import norms, activations
from ..utils.torch_utils import pad_x_to_y
from ..utils.deprecation_utils import VisibleDeprecationWarning
import warnings


class DeMask(BaseEncoderMaskerDecoder):
    """
    Simple MLP model for surgical mask speech enhancement A transformed-domain masking approach is used.

    Args:
        input_type (str, optional): whether the magnitude spectrogram "mag" or both real imaginary parts "reim" are
                    passed as features to the masker network.
                    Concatenation of "mag" and "reim" also can be used by using "cat".
        output_type (str, optional): whether the masker ouputs a mask
                    for magnitude spectrogram "mag" or both real imaginary parts "reim".

        hidden_dims (list, optional): list of MLP hidden layer sizes.
        dropout (float, optional): dropout probability.
        activation (str, optional): type of activation used in hidden MLP layers.
        mask_act (str, optional): Which non-linear function to generate mask.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.

        fb_name (str): type of analysis and synthesis filterbanks used,
                            choose between ["stft", "free", "analytic_free"].
        n_filters (int): number of filters in the analysis and synthesis filterbanks.
        stride (int): filterbank filters stride.
        kernel_size (int): length of filters in the filterbank.
        encoder_activation (str)
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.
    """

    def __init__(
        self,
        input_type="mag",
        output_type="mag",
        hidden_dims=(1024,),
        dropout=0.0,
        activation="relu",
        mask_act="relu",
        norm_type="gLN",
        fb_name="stft",
        n_filters=512,
        stride=256,
        kernel_size=512,
        sample_rate=16000,
        **fb_kwargs,
    ):
        fb_type = fb_kwargs.pop("fb_type", None)
        if fb_type:
            warnings.warn(
                "Using `fb_type` keyword argument is deprecated and "
                "will be removed in v0.4.0. Use `fb_name` instead.",
                VisibleDeprecationWarning,
            )
            fb_name = fb_type
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )

        n_masker_in = self._get_n_feats_input(input_type, encoder.n_feats_out)
        n_masker_out = self._get_n_feats_output(output_type, encoder.n_feats_out)
        masker = build_demask_masker(
            n_masker_in,
            n_masker_out,
            norm_type=norm_type,
            activation=activation,
            hidden_dims=hidden_dims,
            dropout=dropout,
            mask_act=mask_act,
        )
        super().__init__(encoder, masker, decoder)

        self.input_type = input_type
        self.output_type = output_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.mask_act = mask_act
        self.norm_type = norm_type

    def _get_n_feats_input(self, input_type, encoder_n_out):
        if input_type == "reim":
            return encoder_n_out

        if input_type not in {"mag", "cat"}:
            raise NotImplementedError("Input type should be either mag, reim or cat")

        n_feats_input = encoder_n_out // 2
        if input_type == "cat":
            n_feats_input += encoder_n_out
        return n_feats_input

    def _get_n_feats_output(self, output_type, encoder_n_out):
        if output_type == "mag":
            return encoder_n_out // 2
        if output_type == "reim":
            return encoder_n_out
        raise NotImplementedError("Output type should be either mag or reim")

    def forward_masker(self, tf_rep):
        """Estimates masks based on time-frequency representations.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in
                (batch, freq, seq).

        Returns:
            torch.Tensor: Estimated masks in (batch, freq, seq).
        """
        masker_input = tf_rep
        if self.input_type == "mag":
            masker_input = mag(masker_input)
        elif self.input_type == "cat":
            masker_input = magreim(masker_input)
        est_masks = self.masker(masker_input)
        if self.output_type == "mag":
            est_masks = est_masks.repeat(1, 2, 1)
        return est_masks

    def apply_masks(self, tf_rep, est_masks):
        """Applies masks to time-frequency representations.

        Args:
            tf_rep (torch.Tensor): Time-frequency representations in
                (batch, freq, seq).
            est_masks (torch.Tensor): Estimated masks in (batch, freq, seq).

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        if self.output_type == "reim":
            tf_rep = tf_rep.unsqueeze(1)
        return est_masks * tf_rep

    def get_model_args(self):
        """ Arguments needed to re-instantiate the model. """
        model_args = {
            "input_type": self.input_type,
            "output_type": self.output_type,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "activation": self.activation,
            "mask_act": self.mask_act,
            "norm_type": self.norm_type,
        }
        model_args.update(self.encoder.filterbank.get_config())
        return model_args


def build_demask_masker(
    n_in,
    n_out,
    activation="relu",
    dropout=0.0,
    hidden_dims=(1024,),
    mask_act="relu",
    norm_type="gLN",
):
    make_layer_norm = norms.get(norm_type)
    net = [make_layer_norm(n_in)]
    layer_activation = activations.get(activation)()
    in_chan = n_in
    for hidden_dim in hidden_dims:
        net.extend(
            [
                nn.Conv1d(in_chan, hidden_dim, 1),
                make_layer_norm(hidden_dim),
                layer_activation,
                nn.Dropout(dropout),
            ]
        )
        in_chan = hidden_dim

    net.extend([nn.Conv1d(in_chan, n_out, 1), activations.get(mask_act)()])
    return nn.Sequential(*net)
