# import numpy as np
import cv2
import os

dataset = []
dir = './code/AI/AI_final/Human-pose-estimation/cats_dogs/train_data/'
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    for image in os.listdir(file_path):
        img = cv2.imread(os.path.join(file_path, image), cv2.IMREAD_COLOR)
        dataset.append([img, file])
print(dataset)


