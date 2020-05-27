import os

from .data import Signal, get_frames
from .frames import input_face_embeddings
from .audio_feature_generator import convert_to_spectrogram, convert_to_wave

STORAGE_DIR = os.environ.get("STORAGE_DIR", "storage_dir")

