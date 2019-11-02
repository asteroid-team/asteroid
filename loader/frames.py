from PIL import Image
import torch
import numpy as np
from typing import Union, List
from memory_profiler import profile
from facenet_pytorch import MTCNN, InceptionResnetV1

@profile
def input_face_embeddings(frames: Union[List[str], np.ndarray], is_path: bool,
                         mtcnn: MTCNN, resnet: InceptionResnetV1,
                         face_embed_cuda: bool, use_half: bool) -> torch.Tensor:
    """
        Get the face embedding

        NOTE: If a face is not detected by the detector, 
        instead of throwing an error it zeros the input 
        for embedder.

        NOTE: Memory hungry function, hence the profiler.

        Args:
            frames: Frames from the video
            is_path: Whether to read from filesystem or memory
            mtcnn: face detector
            resnet: face embedder
            face_embed_cuda: use cuda for model
            use_half: use half precision

        Returns:
            emb: Embedding for all input frames
    """
    if face_embed_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    result_cropped_tensors = []
    for f in frames:
        if is_path:
            frame = Image.open(f)
        else:
            frame = Image.fromarray(f.astype("uint8"))

        with torch.no_grad():
            cropped_tensors = mtcnn(frame)

        if cropped_tensors is None:
            #Face not detected, for some reason
            cropped_tensors = torch.zeros((3, 160, 160)).to(device)
        elif cropped_tensors.shape[0] == 1:
            #Squeeze the dimenstion if there is only one face
            cropped_tensors = cropped_tensors.squeeze(0).to(device)
        else:
            #Pick a face here
            cropped_tensors = cropped_tensors[0].to(device)

        result_cropped_tensors.append(cropped_tensors)

    del frames
    #Stack all frames
    result_cropped_tensors = torch.stack(result_cropped_tensors)
    #Embed all frames
    result_cropped_tensors = result_cropped_tensors.to(device)
    if use_half:
        result_cropped_tensors = result_cropped_tensors.half()

    with torch.no_grad():
        emb = resnet(result_cropped_tensors)
    if use_half:
        emb = emb.float()
    return emb


if __name__ == '__main__':
    mtcnn = MTCNN(keep_all=True).eval()
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    device = torch.device("cpu")
    res = input_face_embeddings(["a.jpg","b.jpg"], True, mtcnn, resnet, device)
    print(res.shape) # 512D
    print("Passed")
