
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision as tv
import pandas as pd
import pickle

#скопировать файл model.py в основную папку колаб
from model import Model

"""#Обучение модели LeNet
Обучение модели выполнено на основе модуля, выложенного на Git
https://github.com/ChawDoe/LeNet5-MNIST-PyTorch

В исходном файле train.py были изменены:    
1. пути к train и test датасетам   
root='/content/LeNet-5-MNIST-PyTorch/train'   
root='/content/LeNet-5-MNIST-PyTorch/test'
2. запись обученной модели в файл - оставила только самую последюю модель и именила функцию записи, т.к. на исходный порядок записи колаб ругался, а потом не хотел загружать модель.  
3. Файл был переименован в train_mine.py

Нужно затем перезаписать этот файл в корневой папке колаб.

Копирую датасет для обучения модели в текущую папку колаба
"""

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# 
# git clone https://github.com/ChawDoe/LeNet-5-MNIST-PyTorch.git
# 
#

#переходим в дирректорию LeNet5-MNIST-PyTorch
#!cd /content/LeNet-5-MNIST-PyTorch
#!ls

"""Специально выбрала вариант использования готовых скриптов, а не писала его сама чтобы потренироваться в применении готовых модулей.    
Запускаю измененный файл обучения модели
"""

!python /content/LeNet-5-MNIST-PyTorch/train_mine.py

"""Чтобы в дальнейшем не запускать каждый раз модель на обучение сохранила файл с весами модели с лучшим значением accuracy.     
Дальше его загружала и обучала шум.
"""

# на всякий случай - для удаления лишних папок
#!rm -rf LeNet-5-MNIST-PyTorch

"""#Загрузка датасета для предсказания и обучения шума"""

batch_size = 10 #чтобы можно было выбирать разные картинки для их зашумления

train_dataset = tv.datasets.MNIST('.', train = True, transform = tv.transforms.ToTensor(), download = True)
test_dataset = tv.datasets.MNIST('.', train = False, transform = tv.transforms.ToTensor(), download = True)
train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
test = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)

""" вариант обращения к объектам dataloaderа
dataiter = iter(test)
images, labels = next(dataiter)
print(images.shape)
print(labels.shape)

#Посмотреть что внутри
c = 0 #счетчик
num_img = 5 #номер картинки, которую хотим посмотреть
for X0, y0 in train:
  c += 1
  if c == num_img:
    X = torch.unsqueeze(X0[c], dim = 0) #сохраняем текущее изображение и добавляем измерение батча для модели
    y = torch.unsqueeze(y0[c], dim = 0)
    plt.imshow(X[0,0,...])
    print(f'истинная цифра - {y[0]}')
    break

#посмотреть размерности данных
X.shape

y.shape

X0.shape

y0.shape

"""#Предсказание"""

device = 'cuda' if torch.cuda.is_available() else 'cpu' #колаб не всегда позволят подключиться к cuda
device

#инициализируем модель той же архитектуры как и обученная (из того же модуля)
model = Model().to(device)

#загружаем сохраненные данные (состояния) обученной модели
#state_model = torch.load('/content/mnist_0.985.pth', weights_only=True)
#state_model = torch.load('/content/mnist_0.985.pth', weights_only=True) #если cuda в колаб доступна
state_model = torch.load('/content/mnist_0.985.pth', weights_only=True, map_location=torch.device('cpu')) #если cuda в колаб недоступна

#передача загруженных состояний модели
model.load_state_dict(state_model)

predict = model(X.to(device)) #предсказание класса для выбранной картинки

predict #посмотреть что получилось

predict = torch.argmax(predict.detach(), dim = -1)
predict.item()

"""#Тестовая генерация шума, посмотреть как складывается с картинкой"""

mu =0 #среднее
sigma = 0.1 #станд. отклонение

noise = np.random.normal(loc = mu, scale = sigma, size = X[0,0,...].shape)

plt.imshow(noise)

X_noise = X + torch.from_numpy(noise)

fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
ax[0].imshow(X[0,0,...])
ax[1].imshow(X_noise[0,0,...])
plt.show()

"""#Обучение шума"""

n=200 #количество итераций изменения шума

model.eval() #переводим модель в режим предсказания (не меняем градиенты, веса и т.д.)

#начальное значение среднего и станд. отклон шума, указываем, что нужно сохранять градиенты, указываем куда перенести (cuda/cpu)
mu = torch.tensor([0.0], requires_grad=True, device = device)
sigma = torch.tensor([0.0], requires_grad=True, device = device)

#Для входых данных указываем нужный тип данных и тип устройства
X = X.type(torch.float32).to(device)
y = y.type(torch.float32).to(device)

optimizer = torch.optim.Adam(params=[mu, sigma], lr=0.01)
loss_func = torch.nn.MSELoss() #среднеквадр ошибка

for i in range(n):
  noise = (mu + sigma * torch.randn_like(X[0,0,...])).to(device) #гененрируем шум
  X_noise = X + noise #прибавляем шум к исходному изображению

  optimizer.zero_grad() #обнуляем градиенты

  pred_y = model(X_noise) #получаем предсказание от зашумленного изображения

  loss = loss_func(-pred_y, y) #считаем разницу между предсказанным классом c "-" (антиградиент) и фактическим

  loss.backward() #вычисление градиентов

  optimizer.step() #шаг оптимизатора

  pred_class = torch.argmax(pred_y.detach(), dim = -1)

  #если уже не верный класс, то останавливаемся
  if torch.argmax(pred_y.detach(), dim = -1) != y:
    print(f'Истинный класс - {y.item()}, Предсказанный класс - {pred_class.item()}')
    break

print(f'итерация - {i}')
print('loss - {:.3f}, класс - {:.0f}, mu - {:.3f}, sigma - {:.3f}'.format(loss.item(), pred_class.item(), mu.item(), sigma.item()))
print('Слева ИСХОДНОЕ изображение, справа ЗАШУМЛЕННОЕ изображение')

fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
ax[0].imshow(X[0,0,...].cpu().detach().numpy())
ax[1].imshow(X_noise[0,0,...].cpu().detach().numpy())
plt.show()