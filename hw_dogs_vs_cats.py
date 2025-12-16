"""
# Dogs vs Cats
**Задание: Обучить модель классификации изображение на 2 класса.    
Задание засчитывается при значениях метрики Log Loss меньше 0.3.**

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16

print(tf.__version__)
print(tf.executing_eagerly())

from matplotlib import pyplot as plt

"""## Функции загрузки данных"""

import os
from random import shuffle
from glob import glob

IMG_SIZE = (224, 224)  # размер входного изображения сети

train_files = glob(r'D:\Mary\Netology\12_Comp_vew_NN\12_6_NN/train/*.jpg')
test_files = glob(r'D:\Mary\Netology\12_Comp_vew_NN\12_6_NN/test/*.jpg')

# загружаем входное изображение и предобрабатываем
def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path)[...,::-1]
    img = cv2.resize(img, target_size)
    return vgg16.preprocess_input(img)  # предобработка для VGG16

# функция-генератор загрузки обучающих данных с диска
def fit_generator(files, batch_size=32):
    batch_size = min(batch_size, len(files))
    while True:
        shuffle(files)
        for k in range(len(files) // batch_size):
            i = k * batch_size
            j = i + batch_size
            if j > len(files):
                j = - j % len(files)
            x = np.array([load_image(path) for path in files[i:j]])
            y = np.array([1. if os.path.basename(path).startswith('dog') else 0.
                          for path in files[i:j]])
            yield (x, y)

# функция-генератор загрузки тестовых изображений с диска
def predict_generator(files):
    while True:
        for path in files:
            yield np.array([load_image(path)]),

len(test_files)

#посмотреть на одно изображение (изображение синее, т.к. не изменен порядок каналов)
img1 = cv2.imread(train_files[0])
plt.imshow(img1)
plt.show()

img1.shape #посмотреть форму одного ихбражения

"""## Визуализируем примеры для обучения"""

fig = plt.figure(figsize=(16, 8))
for i, path in enumerate(train_files[:10], 1):
    subplot = fig.add_subplot(2, 5, i)
    subplot.set_title('%s' % path.split('/')[-1])
    img = cv2.imread(path)[...,::-1] #bgr->rgb изменение порядка каналов
    img = cv2.resize(img, IMG_SIZE)
    plt.imshow(img)
plt.show()

"""## Загружаем предобученную модель"""

# base_model - объект класса keras.models.Model (Functional Model)
base_model = vgg16.VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

base_model.summary()

"""## Добавляем полносвязный слой"""

# фиксируем все веса предобученной сети
#в исходной сети достаточно сверточных слоев для извлечения признаков из изображения
#добавлю полносвязные слои по типу исходной архитектуры vgg16 + batch norm
for layer in base_model.layers:
    layer.trainable = False

x = base_model.layers[-5].output #размер (14, 14, 512)
x = tf.keras.layers.Flatten()(x) #вытягиваем матрицу в вектор
x = tf.keras.layers.Dense(32,
                          activation = 'relu')(x)
x = tf.keras.layers.Dense(32,
                          activation = 'relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(32,
                          activation = 'relu')(x)
x = tf.keras.layers.Dense(1,  # один выход (бинарная классификация)
                          activation='sigmoid',  # функция активации
                          kernel_regularizer=tf.keras.regularizers.l1(1e-4))(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x, name='dogs_vs_cats')

"""## Выводим архитектуру модели"""

model.summary()

"""## Компилируем модель и запускаем обучение"""

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # функция потерь binary_crossentropy (log loss
              metrics=['accuracy'])

val_samples = 5  # число изображений в валидационной выборке

shuffle(train_files)  # перемешиваем обучающую выборку
validation_data = next(fit_generator(train_files[:val_samples], val_samples))
train_data = fit_generator(train_files[val_samples:])  # данные читаем функцией-генератором

# запускаем процесс обучения
model.fit(train_data,
          steps_per_epoch=10,  # число вызовов генератора за эпоху
          epochs=100,  # число эпох обучения
          validation_data=validation_data)

model.save('cats-dogs-vgg16_0506.hdf5')

from keras.models import load_model

model = load_model('cats-dogs-vgg16_0306.hdf5') #загружаю сохраненную модель

"""## Предсказания на проверочной выборке"""

test_pred = model.predict(
    predict_generator(test_files), steps=len(test_files))

fig = plt.figure(figsize=(16, 8))
for i, (path, score) in enumerate(zip(test_files[:10], test_pred[:10]), 1):
    subplot = fig.add_subplot(2, 5, i)
    subplot.set_title('%.2f %s' % (score, os.path.basename(path)))
    img = cv2.imread(path)[...,::-1]
    img = cv2.resize(img, IMG_SIZE)
    subplot.imshow(img)
plt.show()

"""## Готовим данные для отправки"""

import re

with open('submit.csv', 'w') as dst:
    dst.write('id,label\n')
    for path, score in zip(test_files, test_pred):
        dst.write('%s,%f\n' % (re.search('(\d+).jpg$', path).group(1), score))

"""Итоговое значение функции потерь получается со страницы     
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition   
по результатам сформированного файла с предсказаниями

Промежуточная модель:  
Score: 0.44183 c 2-мя дополнительными Dense16+relu, Dense32+relu     
**Итоговая модель:  
Score: 0.14201  
(добавлено 2 * Dense32+relu + batch + Dense32+relu)**
"""
