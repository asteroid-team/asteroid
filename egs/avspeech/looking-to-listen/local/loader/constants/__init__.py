import os

STORAGE_DIR = os.environ.get("STORAGE_DIR", "storage_dir")
if not STORAGE_DIR.startswith("/"):
    STORAGE_DIR = os.path.join("../..", STORAGE_DIR)  # We are in local/loader

AUDIO_MIX_COMMAND_PREFIX = "ffmpeg -y -t 00:00:03 -ac 1 "

AUDIO_DIR = f"{STORAGE_DIR}/storage/audio"
VIDEO_DIR = f"{STORAGE_DIR}/storage/video"
EMBED_DIR = f"{STORAGE_DIR}/storage/embed"
MIXED_AUDIO_DIR = f"{STORAGE_DIR}/storage/mixed"
SPEC_DIR = f"{STORAGE_DIR}/storage/spec"

AUDIO_SET_DIR = f"{STORAGE_DIR}/audio_set/audio"

STORAGE_LIMIT = 5_000_000_000
