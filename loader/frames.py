from PIL import Image
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from memory_profiler import profile

@profile
def input_face_embeddings(frames, is_path, mtcnn, resnet, device):
    result_cropped_tensors = []
    for f in frames:
        if is_path:
            frame = Image.open(f)
        else:
            frame = Image.fromarray(f.astype("uint8"))
        cropped_tensors = mtcnn(frame)
        if cropped_tensors is None:
            cropped_tensors = torch.zeros((3, 160, 160)).to(device)
        elif cropped_tensors.shape[0] == 1:
            cropped_tensors = cropped_tensors.squeeze(0).to(device)
        else:
            #Pick a face here
            cropped_tensors = cropped_tensors[0].to(device)
        result_cropped_tensors.append(cropped_tensors)
    del frames
    result_cropped_tensors = torch.stack(result_cropped_tensors)
    print(result_cropped_tensors.shape)
    emb = resnet(result_cropped_tensors.to(device))
    return emb


if __name__ == '__main__':
    mtcnn = MTCNN(keep_all=True).eval()
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    device = torch.device("cpu")
    res = input_face_embeddings(["a.jpg","b.jpg"], True, mtcnn, resnet, device)
    print(res.shape) # 512D
    print("Passed")
