from torch import nn
from .base_models import BaseEncoderMaskerDecoder
from ..filterbanks import make_enc_dec
from ..filterbanks.transforms import take_mag, take_cat
from ..masknn import norms, activations
from ..utils.torch_utils import pad_x_to_y


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
        hidden_dims=[1024],
        dropout=0.0,
        activation="relu",
        mask_act="relu",
        norm_type="gLN",
        fb_type="stft",
        n_filters=512,
        stride=256,
        kernel_size=512,
        sample_rate=16000,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_type,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )

        n_masker_in = self._get_n_feats_input(input_type, encoder)
        n_masker_out = self._get_n_feats_output(output_type, encoder)
        net = _build_masker_nn(
            n_masker_in,
            n_masker_out,
            norm_type=norm_type,
            activation=activation,
            hidden_dims=hidden_dims,
            dropout=dropout,
            mask_act=mask_act,
        )
        masker = nn.Sequential(*net)
        super().__init__(encoder, masker, decoder)

        self.input_type = input_type
        self.output_type = output_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.mask_act = mask_act
        self.norm_type = norm_type
        self.fb_type = fb_type
        self.n_filters = n_filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.fb_kwargs = fb_kwargs
        self._sample_rate = sample_rate

    def _get_n_feats_input(self, input_type, encoder):
        if input_type == "reim":
            return encoder.n_feats_out

        if input_type not in {"mag", "cat"}:
            raise NotImplementedError("Input type should be either mag, reim or cat")

        n_feats_input = encoder.n_feats_out // 2
        if input_type == "cat":
            n_feats_input += encoder.n_feats_out
        return n_feats_input

    def _get_n_feats_output(self, output_type, encoder):
        if output_type == "mag":
            return encoder.n_feats_out // 2
        if output_type == "reim":
            return encoder.n_feats_out
        raise NotImplementedError("Output type should be either mag or reim")

    def preprocess_masker_input(self, tf_rep):
        """Preprocesses time-frequency representation for mask estimation.

        The time-frenquency representation at the output of the encoder is
        processed before being passed to the masker. The type of processing
        depends on the value of :attr:`input_type`.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation given by the
                encoder

        Returns:
            torch.Tensor: Data to be given as input for mask estimation

        """
        if self.input_type == "mag":
            return take_mag(tf_rep)
        if self.input_type == "cat":
            return take_cat(tf_rep)
        # No need for NotImplementedError as input_type checked at init
        return tf_rep

    def preprocess_product_input(self, tf_rep):
        """Preprocesses time-frequency representation before applying mask.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation given by the
                encoder

        Returns:
            torch.Tensor (torch.Tensor): Data onto which estimated masks are
                applied
        """
        if self.output_type == "mag":
            return tf_rep
        return tf_rep.unsqueeze(1)

    def postprocess_masks(self, est_masks):
        """Postprocesses estimated masks, before applying them.

        Args:
            est_masks (torch.Tensor): Estimated masks

        Returns:
            torch.Tensor: Postprocessed masks.
        """
        if self.output_type == "mag":
            return est_masks.repeat(1, 2, 1)
        # No need for invalid output_types as checked at init
        return est_masks

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
            "fb_type": self.fb_type,
            "n_filters": self.n_filters,
            "stride": self.stride,
            "kernel_size": self.kernel_size,
            "fb_kwargs": self.fb_kwargs,
            "sample_rate": self._sample_rate,
        }
        model_args.update(self.fb_kwargs)
        return model_args


def _build_masker_nn(
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
    return net
