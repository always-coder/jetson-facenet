from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import time
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size = 160, margin = 0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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

def read_image():
    image = Image.open('./test.jpg')
    
    img_aligned, prob = mtcnn(image, return_prob=True)
    aligned = []
    aligned.append(img_aligned)
    aligned = torch.stack(aligned).to(device)
    curr_img_embedding = resnet(aligned).detach().cpu()

    return curr_img_embedding

def main():
     database_name, database_embedding = process_image_database()
     start = time.time()
     img_embedding = read_image()
     end = time.time()

     print("time is:", end-start)
     min_dist = 1.0
     curr_name = 'unknown'
     for i in range(len(database_embedding)):
         dist = (img_embedding - database_embedding[i]).norm().item()         
         if dist < min_dist:
             min_dist = dist
             curr_name = database_name[i]

     print(curr_name)

if __name__ == '__main__':
    main()
