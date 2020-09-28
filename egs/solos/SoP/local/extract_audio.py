from flerken.video.utils import apply_tree, apply_single


def extract_audio(src, dst):
    rep = apply_tree(src, dst,
                     output_options=['-ac', '1', '-ar', '11025'],
                     multiprocessing=0,
                     ext='.mp3',
                     fn=apply_single)
    return rep
