from torchtree import Directory_Tree

import os
from tqdm import tqdm

from flerken.video.utils import apply_single


def extract_frames(VIDEOS_PATH, DST_PATH):
    tree = Directory_Tree(VIDEOS_PATH)
    if not os.path.exists(DST_PATH):
        os.mkdir(DST_PATH)
    tree.clone_tree(DST_PATH)

    list_of_video_paths = list(tree.paths(root=VIDEOS_PATH))
    progress_bar = tqdm(list_of_video_paths)

    for video_path in progress_bar:
        dst_path = video_path.replace(VIDEOS_PATH, DST_PATH).replace('.mp4', '')
        os.mkdir(dst_path)
        dst_path = dst_path + '/%06d.png'
        key = video_path.split('/')[-1].split('.')[0]
        cat = video_path.split('/')[-2]

        input_options = ['-y', '-hide_banner', '-loglevel', 'panic']
        output_options = ['-r', '8', '-s', '256x256']
        apply_single(video_path, dst_path, input_options, output_options, ext=None)
        progress_bar.set_postfix_str(f'{cat}-->{key}')
