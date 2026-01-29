import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from google.colab import drive
import glob
import random
import time



"""#Выбор отдельных кадров из видео для разметки дополнительных кадров с хозяевами"""

file_path = r'/content/VID_20260107_104127.mp4'

jpg_path = r'/content/jpg_mark/'

cap = cv2.VideoCapture(file_path) #открываем видео из которого будем вырезать кадры
mult = 1 #сохранять будем каждый 10-й, потом выберу совсем подходящие
fr = 1 # задаем счетчик кадров
while True:
  ret, frame = cap.read()  #пока упешно воспроизв-ся видео ret - усп/неусп, frame - текущий кадр
  if not ret:
    break #если кадра нет, то выходим из цикла
  if fr % mult == 0: #если номер кадра кратен 20
    jpg_file = file_path.split('/')[-1][:-3] #составляем имя jpg фвйла (без расширения)
    cv2.imwrite(f'{jpg_path}{jpg_file}_{fr}.jpg', frame) #записываем кадр в файл jpg с номером индекса

  fr +=1 #прибавляем счетчик кадров

# Освобождение ресурсов и закрытие окон
cap.release()

#архивируем папки с созданными кадрами для скачивания
!zip -r /content/VID_20260107_104127 /content/jpg_mark #архивируем

#удалить все файлы из папки, чтобы не путать кадры из разных видео
!rm -rf /content/jpg_mark/*

"""#Сканирование разметки с картинки и заполнение txt файла разметки для дополнения датасета

рамки разметки желтого цвета.   
желтый цвет R=255, G=255, B=0, но в paint не всегда получается точно такой желтый цвет. Чтобы найти рамки буду брать диапазон значений пикселей в районе желтого цвета     
matpotlib - RGB   
OpenCV - BGR
"""

def frame_search(direct, direct_out):
  """функция поиска рамок разметки в файлах jpg в указанной папке
  На вход: direct - путь к избражению c рамками разметки
           direct_out - путь к папке куда записывать txt файл с координатами рамки
  На выход: ничего, найденные координаты записываются в txt файл по указанному пути
  * класс по умолчанию 3 - Владельцы, т.к. добавляем только фото с разметкой владельцев
  ** для фото без людей разметки нет, txt файл не создается (см. требования yolo)
  """
  image = cv2.imread(direct) #открываем изображение
  #print(direct)
  img_h, img_w = image.shape[:2] #запоминаем размеры текущего изображения с разметкой
  #проходим циклом по всему изображению и
  #ищем координаты верхнего левого и правого нижнего углов прямоуголника желтого цвета
  x_min, y_min = img_w, img_h # заведомо большие координаты  верх лев
  x_max, y_max = 0, 0 #заведомо маленькие координаты - прав нижний
  yellow_pic = False # флаг, что найден желтый пиксель - разметка есть

  for y in range(img_h): # по высоте - у
    for x in range(img_w): #по ширине - х
      if ( 0 <= image[y,x,0] <= 20) and (230<= image[y,x,1] <= 255) and (230<= image[y,x,2] <= 255): #если цвет пикселя попадает в желтый диапазон
        #print(f'нашли желтый пиксель - {x, y}')
        yellow_pic = True #поднимаем флаг, что разметка есть
        if (x_min == img_w) and (y_min == img_h): #если еще не нашли левый верхний угол рамки
          #print('нашли первый угол')
          x_min, y_min = x, y #запоминаем координаты левого верхнего угла
          #print(x_min, y_min)
        elif (x > x_min):
          #print(f'уточняем x_max {x_max}')
          x_max = x #перезаписываем значение коорд х правого нижнего угла
        elif (y > y_min):
          #print(f'уточняем y_max {y_max}')
          y_max = y #перезапоминаем коорд у правого нижнего угла

  if yellow_pic is True: #если в файле есть разметка
    #print(f'нашли желтый пиксель')
    #Пересчет координат в формат разметки yolo (центр и размеры изображения от 0 до 1)
    w = x_max - x_min #ширина изображения
    h = y_max - y_min #высота изображения
    xc = x_min + w/2 #вычисляем координаты центра изображения
    yc = y_min + h/2

    #масштабируем координаты и размер по размеру исходного изображения
    w, xc = w/img_w, xc/img_w
    h, yc = h/img_h, yc/img_h

    #записываем в файл .txt

    txt_path = direct_out + '/' + direct.split('/')[-1][:-3]+'txt' #составляем имя txt файла с таким же названием как и jpg
    #print(f'txt_path - {txt_path}')
    with open(txt_path, 'w') as file:
      file.write(f'2 {xc} {yc} {w} {h}') #записываем класс 3 (владельцы) и координаты в файл

  return

"""#Деление всех добавляемых кадров на train, test, validation    
**Формирование случайного распределения кадров по папкам**
"""

drive.mount("/content/drive") #подключаем гугл диск к колаб

#Формируем список всех добавляемых картинок
add_imgs = glob.glob(f'{r'/content/drive/MyDrive/NN_diplom/without_marks'}/*.jpg')

add_imgs[0:3] #посмотреть

#Проверяем количество, д.б.323 шт
len(add_imgs)

#формируем список только из имен файлов без путей
add_name_imgs = [im.split('/')[-1] for im in add_imgs]
add_name_imgs[:3] #посмотреть результат

#перемешиваем полученный список перед делением на train, test, validation
random.shuffle(add_name_imgs)

#делим на train, test, validation в пропорции 60:20:20 (%) 193:65:65 (шт)
#после выполним аугментацию изображений train выборки, после аугментации будет соотношение 75:12,5:12,5
train_imgs = add_name_imgs[:193] #первые 193шт
print(len(train_imgs)) #проверить длину

test_imgs = add_name_imgs[193:258]
print(len(test_imgs))

val_imgs = add_name_imgs[258:]
print(len(val_imgs))

"""**Распределение кадров по папкам, аугментация кадров из train папки, формирование txt файлов с разметкой**"""

with_marks = r'/content/drive/MyDrive/NN_diplom/with_marks' #папка на гугл диске, где лежат изображения с разметкой
without_marks = r'/content/drive/MyDrive/NN_diplom/without_marks' #папка на гугл диске, где лежат изображения без разметки
list_imgs = [train_imgs, test_imgs, val_imgs] #создаем список папок, на которые надо разделить изображения

target_fold = [r'/content/train', r'/content/test', r'/content/val'] #список папок, по которым разносятся изображения

target_labels = [r'/content/train_labels', r'/content/test_labels', r'/content/val_labels'] #список папок, в которые будем сохранять разметку рамок

mir_mark_imgs = r'/content/mir_mark_imgs' #вспомогательная папка для сохранения повернутых изображений с разметкой
flips = [0,1,-1] #набор возможных значений поворотов для аугментации
#0 - поворот по оси х, 1 - по оси у, -1 - по обеим осям

for l, t, tl in zip(list_imgs, target_fold, target_labels):
  for path in l: #для каждого изображения в списке (имя изображения)
    img = cv2.imread(f'{without_marks}/{path}') #считываем текущее изображение без разметки
    #img_name = path.split('/')[-1] #вычленяем только имя файла с расширением
    cv2.imwrite(f'{t}/{path}', img) #сохраняем картинку в нужную папку

    #находим соответствующий файл с разметкой, ищем рамки, пересчитываем координаты по требованиям yolo
    #записываем в соответствующую папку
    frame_search(f'{with_marks}/{path}', f'{tl}')

    #аугментация - поворот изображения случайным образом (только для train)
    if t == r'/content/train': #если разбираем обучающий набор
      #то еще создаем зеркальное изображение
      mark_img = cv2.imread(f'{with_marks}/{path}') #открываем изображение с разметкой
      rand_flip = random.choice(flips) #выбираем случайное значение поворота

      mir_img = cv2.flip(img, rand_flip) #поворачиваем исходное изображение без разметки
      mir_img_mark = cv2.flip(mark_img, rand_flip) #так же поворачиваем исходное изображение с разметкой

      mir_name = path[:-4] #убираем расширение из имени файла
      cv2.imwrite(f'{t}/{mir_name}_mir.jpg', mir_img) #сохраняем в целевую папку повернутое избражение
      cv2.imwrite(f'{mir_mark_imgs}/{mir_name}_mir.jpg', mir_img_mark) #сохраняем в вспомогательную папку повернутое изображение с разметкой

      #находим и записываем координаты разметки
      frame_search(f'{mir_mark_imgs}/{mir_name}_mir.jpg', f'{tl}')

#удалить все файлы из папки при неудачной попытке
!rm -rf /content/train/*
!rm -rf /content/train_labels/*

#архивируем папки с разобранными картинками
!zip -r /content/train /content/train #архивируем
!zip -r /content/test /content/test
!zip -r /content/val /content/val
!zip -r /content/train_labels /content/train_labels
!zip -r /content/test_labels /content/test_labels
!zip -r /content/val_labels /content/val_labels
!zip -r /content/mir_mark_imgs /content/mir_mark_imgs

"""#Визуальная проверка выполнения координат разметки"""

direct_imgs = r'/content/drive/MyDrive/NN_diplom/marked_adds/valid/images' # train test
direct_labels = r'/content/drive/MyDrive/NN_diplom/marked_adds/valid/labels'

for path in glob.glob(f'{direct_imgs}/*.jpg'): #проходим по всем изображениям в папке
    print(path)
    image = cv2.imread(path) #считываем текущее изображение
    img_h, img_w = image.shape[:2] #запоминаем размеры текущего изображения
    txt_path = direct_labels +'/' + path.split('/')[-1][:-3] + 'txt' #формируем название txt файла с разметкой
    if os.path.exists(txt_path):
      with open(txt_path, 'r') as file:
        content = file.read() #считываем содержимое файла
        content = content.split(' ')
        content = [float(x) for x in content] #c(класс), xc, yc, w, h #переводим в число
        xc, yc, w, h = content[1], content[2], content[3], content[4]

      x1 = int((xc-w/2)*img_w) #пересчитываем координаты обратно в размерность изображения
      y1 = int((yc-h/2)*img_h)
      x2 = int((xc+w/2)*img_w)
      y2 = int((yc+h/2)*img_h)

      cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2) #рисуем рамку

    plt.imshow(image[:, :, ::-1]) #меняем каналы местами чтобы выглядело привычно
    plt.show()

"""#Аугментация train датасета"""

direct = r'/content'
aug_direct = r'/content/aug'

flips = [0,1,-1] #набор возможных значений поворотов
#0 - поворот по оси х, 1 - по оси у, -1 - по обеим осям
for path in glob.glob(f'{direct}/*.jpg'): #проходим по всем изображениям в папке
  src = cv2.imread(path) #считываем исходное изображение

  rand_flip = random.choice(flips) #выбираем случайное значение поворота
  mir_src = cv2.flip(src, rand_flip) #поворачиваем исходное изображение

  #гаус не делала т.к. рамки разметки уже не определяются
  #gaus_src = cv2.GaussianBlur(src, (10,10), 5, 5) #создание изображения с добавлением гауссова шума
  #с ядром 10х10, сигма х и у 5 (подобрано эмпирически)

  src_file = path.split('/')[-1][:-4] #составляем имя jpg фвйла (без расширения и точки)

  cv2.imwrite(f'{aug_direct}/{src_file}_mir.jpg', mir_src) #записываем повернутое изобр в файл jpg с индексом mir
  #cv2.imwrite(f'{aug_direct}/{src_file}_gaus.jpg', gaus_src) #записываем зашумленное изобр в файл jpg с индексом gaus