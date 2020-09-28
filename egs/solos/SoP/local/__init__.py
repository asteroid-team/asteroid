import os as _os

from Solos import SOLOS_IDS_PATH

from .extract_audio import extract_audio as _get_audio
from .extract_frames import extract_frames as _get_frames
from .youtubesaver import YouTubeSaver as _YTSaver
from .create_index_files import create_files as _create_files

__all__ = ['prepare_data']


def prepare_data():
    # Download
    DST = _os.path.abspath('./data')
    VIDEOS_PATH = _os.path.join(DST, 'videos')
    AUDIO_PATH = _os.path.join(DST, 'audio')
    FRAMES_PATH = _os.path.join(DST, 'frames')
    CSV_PATH = _os.path.join(DST, 'train.csv')
    if not _os.path.exists(DST):
        _os.mkdir(DST)
    if not _os.path.exists(VIDEOS_PATH):
        _os.mkdir(VIDEOS_PATH)
        saver = _YTSaver()
        saver.from_json(dataset_dir=VIDEOS_PATH, json_path=SOLOS_IDS_PATH)

    if not _os.path.exists(AUDIO_PATH):
        _os.mkdir(AUDIO_PATH)
        _get_audio(VIDEOS_PATH, AUDIO_PATH)

    if not _os.path.exists(FRAMES_PATH):
        _os.mkdir(FRAMES_PATH)
        _get_frames(VIDEOS_PATH, FRAMES_PATH)

    if not _os.path.exists(CSV_PATH):
        _create_files(DST, DST, 8, 80)
