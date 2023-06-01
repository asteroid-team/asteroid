import torch, torchaudio
import argparse
import os
from model import MultiDecoderDPRNN

os.makedirs("outputs", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--wav_file",
    type=str,
    default="",
    help="Path to the wav file to run model inference on.",
)
args = parser.parse_args()

mixture, sample_rate = torchaudio.load(args.wav_file)

model = MultiDecoderDPRNN.from_pretrained("JunzheJosephZhu/MultiDecoderDPRNN").eval()
if torch.cuda.is_available():
    model.cuda()
    mixture = mixture.cuda()
sources_est = model.separate(mixture).cpu()
for i, source in enumerate(sources_est):
    torchaudio.save(f"outputs/{i}.wav", source[None], sample_rate)

print(
    "Thank you for using Multi-Decoder-DPRNN to separate your mixture files. \
    Please support our work by citing our paper: http://www.isle.illinois.edu/speech_web_lg/pubs/2021/zhu2021multi.pdf"
)
