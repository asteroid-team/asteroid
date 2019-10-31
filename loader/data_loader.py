import gc
import os
import torch
import numpy as np
import pandas as pd
from data import Signal
from pathlib import Path
from memory_profiler import profile
from frames import input_face_embeddings
from facenet_pytorch import MTCNN, InceptionResnetV1
from audio_feature_generator import convert_to_spectrogram

class AVDataset(torch.utils.data.Dataset):
    

    def __init__(self, dataset_df_path: Path, video_base_dir: Path, input_df_path: Path,
                input_audio_size=2, use_cuda=False, face_embed_cuda=True, use_half=True):
        """
            
            Args:
                dataset_df_path: path for AVSpeech dataset
                video_base_dir: base directory for all the videos
                input_df_path: path for combination dataset
                input_audio_size: total audio/video inputs
                use_cuda: cuda for the dataset
                face_embed_cuda: cuda for pre-trained models
                use_half: use_half precision for pre-trained models
        """
        self.input_audio_size = input_audio_size

        self.dataset_df = pd.read_csv(dataset_df_path.as_posix())
        self.file_names = self.dataset_df.iloc[:, 0]
        
        #All cropped, pre-processed videos
        self.file_names = [os.path.join(video_base_dir.as_posix(), f + "_final.mp4") 
                        for f in self.file_names]

        self.start_times = self.dataset_df.iloc[:, 1]
        self.end_times = self.dataset_df.iloc[:, 2]

        self.face_x = self.dataset_df.iloc[:, 3]
        self.face_y = self.dataset_df.iloc[:, 4]

        #NOTE: All the above information is not being used anywhere right now.

        #Combination dataset stores, mixed audio, `input_audio_size` inputs
        self.input_df = pd.read_csv(input_df_path.as_posix())

        #Use Cuda for dataset
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
   
        #Use Cuda for pre-trained face processing
        self.face_embed_cuda = face_embed_cuda

        #Use half precision for pre-trained models
        self.use_half = use_half

        #Load pre-trained face processing models
        self.mtcnn = MTCNN(keep_all=True).eval()
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()

        if self.face_embed_cuda:
            device = torch.device("cuda:0")
            self.mtcnn = self.mtcnn.to(device)
            self.resnet = self.resnet.to(device)

            self.mtcnn.device = device

        if self.use_half:
            self.resnet = self.resnet.half()
            #mtcnn doesn't support half precision inputs...


        print(f"MTCNN has {sum(np.prod(i.shape) for i in self.mtcnn.parameters())} parameters")
        print(f"RESNET has {sum(np.prod(i.shape) for i in self.resnet.parameters())} parameters")

    def __len__(self):
        return len(self.input_df)

    @profile
    def __getitem__(self, idx):
        row = self.input_df.iloc[idx, :]
        all_signals = []
        
        for i in range(self.input_audio_size):
            #get audio, video path from combination dataframe
            video_path = row[i]
            audio_path = row[i+self.input_audio_size]

            signal = Signal(video_path, audio_path)
            all_signals.append(signal)

        #input audio signal is the last column.
        mixed_signal = Signal.load_audio(row[-1])

        audio_tensors = []
        video_tensors = []

        for i in range(self.input_audio_size):
            #audio to spectrogram
            spectrogram = convert_to_spectrogram(all_signals[i].get_audio())
            #convert to tensor
            audio_tensors.append(torch.from_numpy(spectrogram))

            #retrieve video frames
            raw_frames = all_signals[i].get_video()
            print(raw_frames.shape)
            
            #NOTE: use_cuda = True, only if VRAM ~ 7+GB, if RAM < 8GB it will not work...
            #run the detector and embedder on raw frames
            embeddings = input_face_embeddings(raw_frames, is_path=False, mtcnn=self.mtcnn, resnet=self.resnet,
                                               face_embed_cuda=self.face_embed_cuda, use_half=self.use_half)
            #clean
            del raw_frames
            video_tensors.append(embeddings)

        # video tensors are expected to be (75,1,1024) (h,w,c)
        # list of video tensors where len(list) == num_person
        # so transpose to be of form video_input = list of video tensors (1024,75,1)
        # we will do
        # for i in range(num_person):
        #   slice out each one , video_input[i] (because this will be of (1024,75,1))

        mixed_signal_tensor = torch.Tensor(convert_to_spectrogram(mixed_signal))  #shape  (257,298,2)
        mixed_signal_tensor = torch.transpose(mixed_signal_tensor,0,2) #shape (2,298,257)  , therefore , 2 channels , height = 298 , width = 257	

        audio_tensors = [a.to(self.device) for a in audio_tensors]
        video_tensors = [a.to(self.device) for a in video_tensors]
        mixed_signal_tensor = mixed_signal_tensor.to(self.device)

        return audio_tensors, video_tensors, mixed_signal_tensor


if __name__ == "__main__":
    dataset = AVDataset(Path("../../data/audio_visual/avspeech_train.csv"),
                      Path("../../data/train/"),
                      Path("temp.csv"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for a, v, m in loader:
        print(len(a), len(v), a[0].shape, v[0].shape, m.shape)
