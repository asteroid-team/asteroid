from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def input_face_embeddings(frames_path):
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    result = []
    for path in frames_path:
        embeddings = []
        frame = Image.open(path)
        cropped_tensors = mtcnn(frame)
        for face in cropped_tensors:
            emb = resnet(face.unsqueeze(0))
            embeddings.append(emb)
        result.append(embeddings)
    return result


if __name__ == '__main__':
    res = input_face_embeddings(["a.jpg","b.jpg"])
    print(res[0][0].shape) # 512D
    print("Passed")
