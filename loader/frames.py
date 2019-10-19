from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
import torch
import numpy as np

def input_face_embeddings(frames, is_path=False, use_cuda=True):
    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    result = []
    for f in frames:
        embeddings = []
        if is_path:
            frame = Image.open(f)
        else:
            frame = Image.fromarray(f.astype("uint8"))
        cropped_tensors = mtcnn(frame)
        if cropped_tensors is None:
            #Apparently trimmed video has few frames without any face -_-
            #Hence, to maintain 75 frames appending zeros...
            result.append(torch.zeros((1, 512)).to(device))
            continue
        for face in cropped_tensors:
            emb = resnet(face.unsqueeze(0).to(device))
            embeddings.append(emb)
        #Training requires one face per photo, during inference face will be selected
        #according to the bounding boxes
        result.append(embeddings[0])
    return result


if __name__ == '__main__':
    res = input_face_embeddings(["a.jpg","b.jpg"], True)
    print(res[0][0].shape) # 512D
    print("Passed")
