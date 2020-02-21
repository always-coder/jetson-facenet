from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import cv2
import os
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size = 160, margin = 0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
loader = transforms.Compose([transforms.ToTensor()])

def collate_fn(x):
    return x[0]

def process_image_database():
    
    dataset = datasets.ImageFolder('./test_images')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=1)

    aligned = []
    names = []

    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    print(names)
    return names, embeddings

def get_camera_capture(name, embedding):
    capture = cv2.VideoCapture(0)

    while 1:
        ret, frame = capture.read()
        if (ret):
            start = time.time()
            cv_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
            cv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            curr_frame = Image.fromarray(cv_frame)
           
            bounding_boxes, _ = mtcnn.detect(curr_frame)

            curr_name = 'unknown'
            if bounding_boxes is not None:
                frame_cropped = curr_frame.crop((
                        bounding_boxes[0][0], bounding_boxes[0][1],
                        bounding_boxes[0][2], bounding_boxes[0][3]))
                
                cv2.rectangle(frame, 
                              (bounding_boxes[0][0], bounding_boxes[0][1]),
                              (bounding_boxes[0][2], bounding_boxes[0][3]),
                              (0, 0, 255), 5)

                aligned = []
                if frame_cropped is not None:
                    frame_cropped = loader(frame_cropped)
                    aligned.append(frame_cropped)
                if len(aligned) == 1:
                    facenet_start = time.time()
                    aligned = torch.stack(aligned).to(device)
                    curr_frame_embedding = resnet(aligned).detach().cpu()
                    facenet_end = time.time()
                    print("facenet time:", facenet_end - facenet_start)
            
                    min_dist = 1.0
                    for i in range(len(embedding)):
                        dist = (curr_frame_embedding - embedding[i]).norm().item()         
                        if dist < min_dist:
                            min_dist = dist
                            curr_name = name[i]
                    print(curr_name)
           
            if bounding_boxes is not None:
                x = int(bounding_boxes[0][0]) - 3
                y = int(bounding_boxes[0][1]) - 3
                cv2.putText(frame, curr_name, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            end = time.time()
            print("frame time:", end-start)
            cv2.imshow("face recognize", frame) 
            if cv2.waitKey(100) & 0xff == ord('q'):
                break;

    capture.release()
    cv2.destroyAllWindows()

def main():
     database_name, database_embedding = process_image_database()
     get_camera_capture(database_name, database_embedding)
     
if __name__ == '__main__':
    main()
