from ._dcunet_architectures import make_unet_encoder_decoder_args

# fmt: off
DCCRN_ARCHITECTURES = {
    "DCCRN-CL": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        [
            (  1,  32, (5, 2), (2, 1), (2, 0)),
            ( 32,  64, (5, 2), (2, 1), (2, 1)),
            ( 64, 128, (5, 2), (2, 1), (2, 0)),
            (128, 256, (5, 2), (2, 1), (2, 1)),
            (256, 256, (5, 2), (2, 1), (2, 0)),
            (256, 256, (5, 2), (2, 1), (2, 1)),
        ],
        # Decoders: auto
        "auto",
    ),
}
