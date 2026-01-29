
!pip install ultralytics

import glob
from google.colab import drive
import os
import time
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

from ultralytics import YOLO

drive.mount("/content/drive") #подключаем гугл диск к колаб

#!mkdir -p dataset #создаем в папке колаба папку dataset, в которую запишем датасет
#!cp -r drive/MyDrive/NN_diplom/test_dataset dataset/ #рекурсивно (все папки и подпапки) копируем датасет в диск колаб

"""#Исследование исходного датасета
Датасет сохранен на гугл диске

Считываем картинки в папке и метки к ним
"""

#img_path = glob.glob(r'/content/*.jpg')
#label_path = glob.glob(r'/content/*.txt')

#img_path = glob.glob(r'/content/drive/MyDrive/NN_diplom/test_dataset/images/*.jpg')
#label_path = glob.glob(r'/content/drive/MyDrive/NN_diplom/test_dataset/labels/*.txt')

img_path = glob.glob(r'/content/drive/MyDrive/NN_diplom/dataset/**/*.jpg', recursive=True)
label_path = glob.glob(r'/content/drive/MyDrive/NN_diplom/src_dataset/**/*.txt', recursive=True)

"""##Проверка наличия пустых файлов без разметки
Согласно документции YOLO для изображений не содержащих объекты допускается пустые txt файлы или полное отсутствие txt файлов разметки. Кадров с отсутствующими файлами разметок нет (количество файлов кадров и файлов разметки совпадает). Проверим наличие 0пустых файлов разметки.
"""

#посмотреть есть ли пустые кадры без людей
k=0 #задаем счетчик количества пустых файлов
empty = []
for label in label_path:
 with open(label, 'rt') as f:
    lbls = f.readlines() #считываем все строки как список в переменную
    if len(lbls) == 0: #если нет записей в файле
      k +=1 #прибавляем счетчик
      empty.append(label)

print(k)

len(label_path)

"""Вывод - пустых кадров без людей нет. В целом это также было видно по одинаковому количеству файлов изображений и аннотаций.

##Подсчет соотношения кадров разных классов и соотношения количества объектов разных классов.
"""

#Подсчет количества кадров с только с людьми, только с подозреваемыми и смешанных
k=0 #счетчик всех изображений для контроля
susp_num=0 #счетчик количества только подозреваемых
pers_num = 0 #счетчик количества только людей
mix_pers = 0 #счетчик людей в смешанных кадрах
mix_susp = 0 #счетчик подозреваемых в смешанных кадрах
suspicion = [] #список кадров только с подозреваемыми
human = [] #список кадров только с людьми
mixed = [] #список кадров и с людьми, и сподозреваемыми
for label in label_path: #проходим по всем файлам с разметкой
  k +=1 #прибавляем счетчик изображений
  with open(label, 'rt') as f: #открываем каждый файл
    lbls = f.readlines() #считываем все строки как список в переменную
    pers = 0 #счетчик людей в текущем файле
    susp = 0 #счетчик подозр в текущем файле

    for s in range(0,len(lbls)): #проходим по списку имеющихся в файле разметок
      obj = lbls[s].replace('\n','') #удаляем перенос строки в конце
      obj = [float(x) for x in obj.split()] #делим строку по пробелам и переводим в список чисел
      #номер класса в разметке стоит на первом месте

      if obj[0] == 0.0: #если человек - класс 0
        pers +=1
      else:
        susp +=1

  if (pers == 0) & (susp != 0): #если только подозреваемые
    susp_num += susp #прибавляем количество подозреваемых
    suspicion.append(label) #запрминаем название кадра
  elif (pers != 0) & (susp == 0): #если только люди
    pers_num += pers #прибавляем количество людей
    human.append(label) #запоминаем название кадра
  else: #если есть и люди, и подозреваемые
    mix_pers += pers #прибавляем количество людей
    mix_susp += susp #прибавляем количество подозреваемых
    mixed.append(label) #запоминаем название кадра

print(f'Всего проверено {k} файлов разметки')

print(f'Количество кадров только с людьми - {len(human)}')
print(f'Количество людей на кадрах только с людьми - {pers_num}')

print(f'Количество кадров только с подозреваемыми - {len(suspicion)}')
print(f'Количество подозреваемых на кадрах только с подозреваемыми - {susp_num}')

print(f'Количество смешанных кадров - {len(mixed)}')
print(f'Количество людей на смешанных кадров - {mix_pers}')
print(f'Количество подозреваемых на смешанных кадрах - {mix_susp}')

print(f'Всего количество людей - {pers_num + mix_pers}')
print(f'Всего количество подозреваемых - {susp_num + mix_susp}')

suspicion[0:3] #росмотреть 1-е 3 таких файла

type(suspicion[0:3])

"""##Визуальный анализ кадров подозрительных объектов
В датасете есть фото лиц людей в масках, фото оружия, на них нет человека, но выделено подозрительное поведение. Наряду с этим, есть фото людей с подозрительным поведением, которые замаркированы как подозрительное поведение
"""

#объединяю в список какие картинки с разметкой посмотреть
just_look = [mixed[0], mixed[1000], mixed[2000], suspicion[0], suspicion[1000], suspicion[2000]]

just_look

just_look = ['/content/drive/MyDrive/NN_diplom/src_dataset/train/labels/stabf13_16_jpg.rf.6be8e7a66ad87eaa9ea2621de325a5a0.txt',
 '/content/drive/MyDrive/NN_diplom/src_dataset/train/labels/stabf13_31_jpg.rf.6ca577e43c0ae01f99fb4d1671aa0530.txt',
 '/content/drive/MyDrive/NN_diplom/src_dataset/train/labels/frame1540_jpg.rf.94b7078e09394f210e902d0c15699f09.txt',
 '/content/drive/MyDrive/NN_diplom/src_dataset/train/labels/rgb-0000050_jpg.rf.a6a78079800194ef1192b0e6b07aea3c.txt',
 '/content/drive/MyDrive/NN_diplom/src_dataset/train/labels/mask-4453-_jpg.rf.ba6eb8b483b81805e01f4154614e73da.txt',
 '/content/drive/MyDrive/NN_diplom/src_dataset/train/labels/mask-2908-_jpg.rf.421a8929aac1ccb7efbd2bc8f2e314f5.txt']

#выборочно посмотреть на картинки  и разметку для подозрительных и смешанных кадров
path = r'/content/' #путь к папке с изображениями и разметкой

for jl in just_look:
  print(jl)
  file_name = jl.split('/')[-1][:-3] #оставляем только название файла без расширения
  img_path = f'{path + file_name}jpg' #формируем имя кадра из имени файла разметки
  img = cv2.imread(img_path) #ищем нужную картинку и открываем
  print(img.shape) #посмотреть размер текущей картинки
  img_h, img_w = img.shape[:2] #запоминаем размеры текущего изображения

  #открываем и считываем рамки объектов из файлов меток
  with open(jl, 'rt') as f:
    lbls = f.readlines() #считываем все строки как список в переменную
    objects = [] #создаем пустой список, в который будем соханять имеющуюся разметку объектов из файла.txt
    for s in range(0,len(lbls)): #проходим по списку имеющихся в файле разметок
      obj = lbls[s].replace('\n','') #удаляем перенос строки в конце
      obj = [float(x) for x in obj.split()] #делим строку по пробелам и переводим в список чисел

      #перевод координат рамок из диапазона 0-1 в диапазон фактического размера изображения
      obj[1] *= img_w #x-центр (ширина) умножаем на ширину изображения
      obj[2] *= img_h #y-центр (высота) умножаем на высоту изображения
      obj[3] *= img_w #ширина (х)
      obj[4] *= img_h #высота(у)
      obj = [int(x) for x in obj] #переводим в целые числа
      #print(obj)

      objects.append(obj) #добавляем к списку объектов

  #рисуем считанные рамки и подписываем класс объекта
  for i in range(0,len(objects)): #проходим по списку объектов
    x1 = int(objects[i][1] - objects[i][3]/2) #левый верхний угол рамки
    y1 = int(objects[i][2] - objects[i][4]/2) #левый верхний угол рамки
    x2 = int(objects[i][1] + objects[i][3]/2) #правый нижний угол рамки
    y2 = int(objects[i][2] + objects[i][4]/2) #правый нижний угол рамки
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2) #рисуем рамки
    cv2.putText(img, str(objects[i][0]),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) #подписываем класс


  #рисуем картинку
  plt.imshow(img[:,:,::-1])
  plt.show()

"""Видно, что картинки могут быть разного размера

##Просмотр кадров с разметкой для класса 0 - люди
"""

imgs = glob.glob(r'/content/*.txt') #составить список аннотаций
path = r'/content/' #путь к папке с изображениями и разметкой
for jl in imgs:
  print(jl)
  file_name = jl.split('/')[-1][:-3] #оставляем только название файла без расширения
  img_path = f'{path + file_name}jpg' #формируем имя кадра из имени файла разметки
  img = cv2.imread(img_path) #ищем нужную картинку и открываем
  print(img.shape) #посмотреть размер текущей картинки
  img_h, img_w = img.shape[:2] #запоминаем размеры текущего изображения

  #открываем и считываем рамки объектов из файлов меток
  with open(jl, 'rt') as f:
    lbls = f.readlines() #считываем все строки как список в переменную
    objects = [] #создаем пустой список, в который будем соханять имеющуюся разметку объектов из файла.txt
    for s in range(0,len(lbls)): #проходим по списку имеющихся в файле разметок
      obj = lbls[s].replace('\n','') #удаляем перенос строки в конце
      obj = [float(x) for x in obj.split()] #делим строку по пробелам и переводим в список чисел

      #перевод координат рамок из диапазона 0-1 в диапазон фактического размера изображения
      obj[1] *= img_w #x-центр (ширина) умножаем на ширину изображения
      obj[2] *= img_h #y-центр (высота) умножаем на высоту изображения
      obj[3] *= img_w #ширина (х)
      obj[4] *= img_h #высота(у)
      obj = [int(x) for x in obj] #переводим в целые числа
      #print(obj)

      objects.append(obj) #добавляем к списку объектов

  #рисуем считанные рамки и подписываем класс объекта
  for i in range(0,len(objects)): #проходим по списку объектов
    x1 = int(objects[i][1] - objects[i][3]/2) #левый верхний угол рамки
    y1 = int(objects[i][2] - objects[i][4]/2) #левый верхний угол рамки
    x2 = int(objects[i][1] + objects[i][3]/2) #правый нижний угол рамки
    y2 = int(objects[i][2] + objects[i][4]/2) #правый нижний угол рамки
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2) #рисуем рамки
    cv2.putText(img, str(objects[i][0]),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) #подписываем класс


  #рисуем картинку
  plt.imshow(img[:,:,::-1])
  plt.show()

"""#Визуализация детекции и истинной разметки на test наборе"""

#детекция объектов с дообученной на 10 эпохах моделью yolo11n
# Загрузка дообученной модели YOLOv11
model_best_yolo_10 = YOLO(r'/content/best_11s_128.pt')

imgs = glob.glob(r'/content/test/*.txt') #составить список аннотаций
path = r'/content/test/' #путь к папке с изображениями и разметкой
for jl in imgs:
  print(jl)
  file_name = jl.split('/')[-1][:-3] #оставляем только название файла без расширения
  img_path = f'{path + file_name}jpg' #формируем имя кадра из имени файла разметки
  img = cv2.imread(img_path) #ищем нужную картинку и открываем
  #print(img.shape) #посмотреть размер текущей картинки
  img_h, img_w = img.shape[:2] #запоминаем размеры текущего изображения

  #детекция объектов на текущем кадре
  #передаем в модель YOLO текущий кадр, verbose=False чтобы не выводилась инфа о работе модели
  result = model_best_yolo_10(img, conf = 0.3, classes = [0, 2], verbose = False)[0]
  #получаем предсказанные моделью координаты и классы
  pred_cls = result.boxes.cls.numpy().astype(np.int32) #класс
  pred_bxs = result.boxes.xyxy.numpy().astype(np.int32) #координаты

  #открываем и считываем рамки объектов из файлов меток
  with open(jl, 'rt') as f:
    lbls = f.readlines() #считываем все строки как список в переменную
    objects = [] #создаем пустой список, в который будем соханять имеющуюся разметку объектов из файла.txt
    for s in range(0,len(lbls)): #проходим по списку имеющихся в файле разметок
      obj = lbls[s].replace('\n','') #удаляем перенос строки в конце
      obj = [float(x) for x in obj.split()] #делим строку по пробелам и переводим в список чисел

      #перевод координат рамок из диапазона 0-1 в диапазон фактического размера изображения
      obj[1] *= img_w #x-центр (ширина) умножаем на ширину изображения
      obj[2] *= img_h #y-центр (высота) умножаем на высоту изображения
      obj[3] *= img_w #ширина (х)
      obj[4] *= img_h #высота(у)
      obj = [int(x) for x in obj] #переводим в целые числа
      #print(obj)

      objects.append(obj) #добавляем к списку объектов

  #рисуем истинные считанные рамки и подписываем класс объекта
  for i in range(0,len(objects)): #проходим по списку объектов
    x1 = int(objects[i][1] - objects[i][3]/2) #левый верхний угол рамки
    y1 = int(objects[i][2] - objects[i][4]/2) #левый верхний угол рамки
    x2 = int(objects[i][1] + objects[i][3]/2) #правый нижний угол рамки
    y2 = int(objects[i][2] + objects[i][4]/2) #правый нижний угол рамки
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2) #рисуем рамки
    cv2.putText(img, str(objects[i][0]),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2) #подписываем класс
  #рисуем предсказанные рамки и класс
  for pr_bxs, pr_cls in zip(pred_bxs, pred_cls): #для всех предсказанных объектов
    x1, y1, x2, y2 = pr_bxs #вычленяем координаты
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 4) #рисуем рамки
    cv2.putText(img, str(pr_cls), (x1, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

  #Сохраняем изображение в файл
  cv2.imwrite(f'res_{file_name}.jpg', img)

  #рисуем картинку
  print('Синий - истинные аннотации, красный - предсказанный')
  plt.imshow(img[:,:,::-1])
  plt.show()

pred_cls