from __future__ import unicode_literals
import json as _json
import os as _os
import sys as _sys
import fire as _fire
import youtube_dl as _youtube_dl

from Solos import SOLOS_IDS_PATH

__all__ = ['YouTubeSaver']


class YouTubeSaver(object):
    """Load video from YouTube using an auditionDataset.json """

    def __init__(self):
        self.outtmpl = '%(id)s.%(ext)s'
        self.ydl_opts = {
            'format': 'bestvideo+bestaudio',
            'outtmpl': self.outtmpl,
            """
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            """
            'logger': None
        }

    def from_json(self, dataset_dir, json_path=SOLOS_IDS_PATH):
        dataset = _json.load(open(json_path))

        for instrument in dataset.keys():
            if not _os.path.exists(_os.path.join(dataset_dir, instrument)):
                _os.makedirs(_os.path.join(dataset_dir, instrument))
            self.ydl_opts['outtmpl'] = _os.path.join(dataset_dir, instrument, self.outtmpl)
            with _youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                for i, video_id in enumerate(dataset[instrument]):
                    try:
                        ydl.download(['https://www.youtube.com/watch?v=%s' % video_id])
                        del dataset[video_id]
                    except KeyboardInterrupt:
                        _sys.exit()
                    except Exception:
                        with open(_os.path.join(dataset_dir, 'backup.json'), 'w') as dst_file:
                            _json.dump(dataset, dst_file)
                        print('Process failed at video {0}, #{1}'.format(video_id, i))
                        print('Backup saved at {0}'.format(_os.path.join(dataset_dir, 'backup.json')))
                        ydl.download(['https://www.youtube.com/watch?v=%s' % video_id])


if __name__ == '__main__':
    _fire.Fire(YouTubeSaver)

    # USAGE
    # python youtubesaver.py from_json /path_to_your_dst
