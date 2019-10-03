import torch
from data import Signal
from pathlib import Path


class AVLoader(torch.utils.data.DataLoader):
    

    def __init__(self, df_path: Path, video_base_dir: Path):
        self.df = pd.read_csv(df_path.as_posix())
        self.file_names = df.iloc[:, 0]
        self.file_names = [os.path.join(video_base_dir.as_posix(), f + "_cropped.mp4") 
                        for f in file_names]
        self.start_times = df[:, 1]
        self.end_times = df[:, 2]

        self.face_x = df[:, 3]
        self.face_y = df[:, 4]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        video_path = self.video_path[idx]
        
        signal = Signal(video_path)
        


