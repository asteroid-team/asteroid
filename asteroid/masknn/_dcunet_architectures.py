from ..utils.generic_utils import unet_decoder_args


def make_unet_encoder_decoder_args(encoder_args, decoder_args):
    encoder_args = tuple(
        (
            in_chan,
            out_chan,
            kernel_size,
            stride,
            [n // 2 for n in kernel_size] if padding == "auto" else padding,
        )
        for in_chan, out_chan, kernel_size, stride, padding in encoder_args
    )

    if decoder_args == "auto":
        decoder_args = unet_decoder_args(
            encoder_args,
            skip_connections=True,
        )
    else:
        decoder_args = tuple(
            (
                in_chan,
                out_chan,
                kernel_size,
                stride,
                [n // 2 for n in kernel_size] if padding == "auto" else padding,
                output_padding,
            )
            for in_chan, out_chan, kernel_size, stride, padding, output_padding in decoder_args
        )

    return encoder_args, decoder_args


# fmt: off

DCUNET_ARCHITECTURES = {
    "DCUNet-10": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            ( 1, 32, (7, 5), (2, 2), "auto"),
            (32, 64, (7, 5), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DCUNet-16": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            ( 1, 32, (7, 5), (2, 2), "auto"),
            (32, 32, (7, 5), (2, 1), "auto"),
            (32, 64, (7, 5), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DCUNet-20": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            ( 1, 32, (7, 1), (1, 1), "auto"),
            (32, 32, (1, 7), (1, 1), "auto"),
            (32, 64, (7, 5), (2, 2), "auto"),
            (64, 64, (7, 5), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 64, (5, 3), (2, 1), "auto"),
            (64, 64, (5, 3), (2, 2), "auto"),
            (64, 90, (5, 3), (2, 1), "auto"),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "Large-DCUNet-20": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            ( 1,  45, (7, 1), (1, 1), "auto"),
            (45,  45, (1, 7), (1, 1), "auto"),
            (45,  90, (7, 5), (2, 2), "auto"),
            (90,  90, (7, 5), (2, 1), "auto"),
            (90,  90, (5, 3), (2, 2), "auto"),
            (90,  90, (5, 3), (2, 1), "auto"),
            (90,  90, (5, 3), (2, 2), "auto"),
            (90,  90, (5, 3), (2, 1), "auto"),
            (90,  90, (5, 3), (2, 2), "auto"),
            (90, 128, (5, 3), (2, 1), "auto"),
        ),
        # Decoders:
        # (in_chan, out_chan, kernel_size, stride, padding, output_padding)
        (
            (128, 90, (5, 3), (2, 1), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 1), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 1), "auto", (0, 0)),
            (180, 90, (5, 3), (2, 2), "auto", (0, 0)),
            (180, 90, (7, 5), (2, 1), "auto", (0, 0)),
            (180, 90, (7, 5), (2, 2), "auto", (0, 0)),
            (135, 90, (1, 7), (1, 1), "auto", (0, 0)),
            (135,  1, (7, 1), (1, 1), "auto", (0, 0)),
        ),
    ),
    "mini": make_unet_encoder_decoder_args(
        # This is a dummy architecture used for Asteroid unit tests.

        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        (
            (1, 4, (7, 5), (2, 2), "auto"),
            (4, 8, (7, 5), (2, 2), "auto"),
            (8, 16, (5, 3), (2, 2), "auto"),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
}
