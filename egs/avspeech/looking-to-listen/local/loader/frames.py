from PIL import Image
import torch
import numpy as np
from typing import Union, List
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

cpu_device = torch.device("cpu")


def input_face_embeddings(
    frames: Union[List[str], np.ndarray],
    is_path: bool,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    face_embed_cuda: bool,
    use_half: bool,
    coord: List,
    name: str = None,
    save_frames: bool = False,
) -> torch.Tensor:
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
    if face_embed_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    result_cropped_tensors = []
    no_face_indices = []
    for i, f in enumerate(frames):
        if is_path:
            frame = Image.open(f)
        else:
            frame = Image.fromarray(f.astype("uint8"))

        with torch.no_grad():
            cropped_tensors = None
            height, width, c = f.shape
            bounding_box, prob = mtcnn.detect(frame)

            if bounding_box is not None:
                for box in bounding_box:
                    x1, y1, x2, y2 = box
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1

                    # for point in coord:
                    x, y = coord[0], coord[1]
                    x *= width
                    y *= height
                    if x >= x1 and y >= y1 and x <= x2 and y <= y2:
                        cropped_tensors = extract_face(frame, box)
                        # print("found", box, x, y, end='\r')
                        break

        if cropped_tensors is None:
            # Face not detected, for some reason
            cropped_tensors = torch.zeros((3, 160, 160))
            no_face_indices.append(i)

        if save_frames:
            name = name.replace(".mp4", "")
            saveimg = cropped_tensors.detach().cpu().numpy().astype("uint8")
            saveimg = np.squeeze(saveimg.transpose(1, 2, 0))
            Image.fromarray(saveimg).save(f"{name}_{i}.png")

        result_cropped_tensors.append(cropped_tensors.to(device))

    if len(no_face_indices) > 20:
        # few videos start with silence, allow 0.5 seconds of silence else remove
        return None
    del frames
    # Stack all frames
    result_cropped_tensors = torch.stack(result_cropped_tensors)
    # Embed all frames
    result_cropped_tensors = result_cropped_tensors.to(device)
    if use_half:
        result_cropped_tensors = result_cropped_tensors.half()

    with torch.no_grad():
        emb = resnet(result_cropped_tensors)
    if use_half:
        emb = emb.float()
    return emb.to(cpu_device)


if __name__ == "__main__":
    mtcnn = MTCNN(keep_all=True).eval()
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    device = torch.device("cpu")
    res = input_face_embeddings(["a.jpg", "b.jpg"], True, mtcnn, resnet, device)
    print(res.shape)  # 512D
    print("Passed")
