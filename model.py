
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

#путь к файлу конфигурации набора дополненных данных
yaml_path = r'/content/drive/MyDrive/NN_diplom/full_dataset/data_add.yaml'

"""#Обучение модели **yolo11s** на изображениях 64x64 с возобновлением обучения"""

#инициализируем модель
Yolo_model11s = YOLO('yolo11s.pt')

Yolo_model11s.info()

#начальные настройки обучения - ниже обучение ниже возобновленное обучение на промежуточных сохраненных параметрах модели
results = Yolo_model11s.train(data = yaml_path, epochs = 10, imgsz = 128, device= device)

#архивируем папку runs с результатами обучения в архив runs_yolo
!zip -r /content/runs_yolo_11s /content/runs
files.download(r'/content/runs_yolo_11s.zip')

"""#Возобновление обучения"""

#возобновление обучение на ранее сохраненных промежуточных контрольных точках
resumed_yolo11s = YOLO(r'/content/yolo11s_64.pt') #загрузила в content последнюю сподсчитанную ранее модель
results = resumed_yolo11s.train(data = yaml_path, epochs = 10, imgsz = 64, device= device, resume=True)
#resume=True - добавить при вызове с перерывом обучения

#архивируем папку runs с результатами обучения в архив runs_yolo
!zip -r /content/runs_yolo_11s /content/runs
files.download(r'/content/runs_yolo_11s.zip')

"""#Обучение yolo11s с параметрами отличными от настроек по умолчанию, но с размером изображения как в обученной выше модели yolo11n  128x128    
patience = 2 - количество эпох ожидания без улучшения метрик проверки   
batch = 32 - попробовать ускорить обучение   
save = True - сохрание промежуточных контрольных точек обучения для возможности возрбновления обучения   
save_period = 1 - сохранять контрольные точки каждую эпоху   
pretrained = True - указываем, что слудет начать обучение с предварительно обученной модели  
classes = [1,2] - только классы люди и владельцы  
dropout = 0.8 - для исключения переобучения
"""

#начальные настройки обучения - ниже обучение ниже возобновленное обучение на промежуточных сохраненных параметрах модели
results = Yolo_model11s.train(data = yaml_path, epochs = 10, imgsz = 128, device= device, patience = 2, batch = 32, save = True, save_period = 1, pretrained = True, classes = [0,2], dropout = 0.8)

#архивируем папку runs с результатами обучения в архив runs_yolo
!zip -r /content/runs_yolo_11s /content/runs
files.download(r'/content/runs_yolo_11s.zip')

"""#Возобновление обучения"""

resumed_yolo11s = YOLO(r'/content/last_8.pt')
results = resumed_yolo11s.train(data = yaml_path, epochs = 10, imgsz = 128, device= device, patience = 2, batch = 32, save = True, pretrained = True, save_period = 1, classes = [0,2], dropout = 0.8, resume=True)


#архивируем папку runs с результатами обучения в архив runs_yolo
!zip -r /content/runs_yolo_11s /content/runs
files.download(r'/content/runs_yolo_11s.zip')

"""1-2 эпохи - 7.5 часов   
3- 4 эпохи -  6 часов    
5-7 эпохи - 8 часов
8-10 - 6.5 часов
Всего - 28 часов
   
"""