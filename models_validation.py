
!pip install ultralytics

from ultralytics import YOLO

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from google.colab import drive
import torch
from google.colab import files

drive.mount("/content/drive") #подключаем гугл диск к колаб

device = 'cuda' if torch.cuda.is_available() else 'cpu' #колаб не всегда позволят подключиться к cuda
device

yaml_path = r'/content/drive/MyDrive/NN_diplom/full_dataset/data_add.yaml'

"""#Валидация на **test** данных недообученной модели yolo11n"""

model_yolo = YOLO('yolo11n.pt')

metrics_yolo = model_yolo.val(data = yaml_path, device = device, split = 'test')

!zip -r /content/runs_yolo /content/runs/detect/val
files.download(r'/content/runs_yolo.zip')

"""#Валидация на **train** данных недообученной модели yolo11n"""

metrics_yolo_train = model_yolo.val(data = yaml_path, device = device, split = 'train')

"""#Валидация на **test** данных дообученной модели yolo11n **10 эпох**"""

model_yolo_10 = YOLO('/content/best.pt')

#валидация дообученной модели  на test данных
metrics_yolo_10 = model_yolo_10.val(data = yaml_path, device = device, split = 'test')

!zip -r /content/runs_yolo_10 /content/runs/detect/val2
files.download(r'/content/runs_yolo_10.zip')

"""#Валидация на **train** данных дообученной модели yolo11n **10 эпох**"""

model_yolo_10 = YOLO('/content/best_10.pt')

metrics_yolo_train_10 = model_yolo_10.val(data = yaml_path, device = device, split = 'train')

"""#Валидация на **test** данных дообученной модели yolo11n **50 эпох**"""

model_yolo_50 = YOLO('/content/best_yolo_50.pt')

#валидация дообученной модели  на test данных
metrics_yolo_50 = model_yolo_50.val(data = yaml_path, device = device, split = 'test')
!zip -r /content/runs_yolo_50 /content/runs/detect/val
files.download(r'/content/runs_yolo_50.zip')

"""#Валидация на **train** данных дообученной модели yolo11n **50 эпох**"""

metrics_yolo_train_50 = model_yolo_50.val(data = yaml_path, device = device, split = 'train')

"""#Валидация на **test** данных дообученной модели **yolo11s** с размером изображения 64х64"""

yolo_11s_64 = YOLO('/content/best_yolo_11s_64.pt')

#валидация дообученной модели  на test данных
metrics_yolo11s_64 = yolo_11s_64.val(data = yaml_path, device = device, split = 'test')
!zip -r /content/runs_yolo11s_64 /content/runs/detect/val
files.download(r'/content/runs_yolo11s_64.zip')

"""#Валидация на **train** данных дообученной модели **yolo11s** 10 эпох c размером иображения 64х64"""

metrics_yolo11s_train = yolo_11s_64.val(data = yaml_path, device = device, split = 'train')

"""#Валидация на **test** данных дообученной модели **yolo11s** с размером изображения 128х128"""

yolo_11s_128 = YOLO('/content/best_11s_128.pt')

#валидация дообученной модели  на test данных
metrics_yolo11s_128 = yolo_11s_128.val(data = yaml_path, device = device, split = 'test')

"""#Валидация на **train** данных дообученной модели **yolo11s** 10 эпох c размером иображения 128х128"""

metrics_yolo11s_train = yolo_11s_128.val(data = yaml_path, device = device, split = 'train')