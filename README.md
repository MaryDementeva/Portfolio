<img src="фото для резюме.jpg" align="left" width="110" height="120" />

**Дементьева Мария Александровна**<br/>
**1979 г.р.**<br/>
email: mbrovarova@mail.ru<br/>
тел:+7 905 305 8017

<br/>
<br/>
В настоящее время заканчиваю обучение на курсе <ins>Data Scientist: расширенная траектория</ins> в образовательной онлайн-платформе Нетология. <br/>  

[Ссылка на программу курса](https://netology.ru/programs/prodatascience?programName=data-scientist#/modul_2)

Параллельно работаю ведущим специалистом в проектно-строительной организации.<br/>

### Тестовые задания<br/>
#### <ins>Построение классификатора изображений на основе предобученной нейронной сети</ins>.  
По заданию требовалось построить бинарнай классификатор изображений. При этом значение функции потерь должно быть не более 0,3.<br/>
Использовался Python.  
Основные этапы:   
- загрузка изображений (glob), разделение на обучающую и тестовую выборки, 
- загрузка предобученной модели vgg16 без полносвязных слоев (tensorflow.keras), 
- последовательный подбор полносвязных слоев для достижения требуемого значения функции потерь (tensorflow.keras).

Задание выполнено в [файле](test_projects/HW_dogs_vs_cats.ipynb). 

#### <ins>Выбор метода для оценки различий между группами: обработка результатов A/B тестов</ins>.  
Использовался Python.  
Основные этапы:   
- анализ данных, построение графиков, расчет статистических параметров (scipy), 
- подбор критерия в зависимости от типа распределения (scipy), 
- интерпритация результатов.
  
Задание выполнено в [файле](test_projects/АВ_test.ipynb).

### Самостоятельно выполненные проекты вне рамок обучения<br/>
#### <ins>Кластеризация игроков в покер в зависимости от их поведения во время игры с целью определения наилучшей стратегии</ins>.  
Использовался Python.  
Действия игроков записаны в текстовых файлах в определенном формате.   
Разбор действий выполнялся по определенным правилам, не обозначенным в файле по просьбе заказчика. Анализ полученных результатов также не показан в файле по просьбе заказчика.  

Основные этапы:   
- считывание и разбор по правилам данных из текстовых файлов (glob, re), 
- работа с датафреймами (pandas), 
- формирование новых параметров для улучшения метрик кластеризации, 
- кластеризация игроков несколькими методами для выбора оптимального (sklearn),
- расчет метрик, понижение размерности для визуализации полученной разбивки (sklearn, sns, matplotlib).

Выборка данных из текстовых файлов и их компоновка в датафрейм выполняется в [файле](poker/poker.ipynb).  
Кластеризация выполняется в [файле](poker/poker_clster.ipynb).

#### <ins>Автоматический разбор присланных за день штрафов, с сортировкой по правилам и рассылкой на почту</ins>.   
Использовался Python.
Основные этапы: 
- выборка файлов по дате создания (sys, os, glob, datetime);
- чтение pdf файлов и выбор из них данных (данные на автомобиль, сумма штрафа, фото номера) (PyPDF2, re);
- запись выбранных данных в эксель файл (openpyxl);
- отправка на почту (smtplib, email.message).

Разбор файлов, формирование отчетов и отправка отчета на почту в [файле](drivers/drivers.ipynb). 

### Некоторые курсовые работы, выполненные в рамках обучения
#### <ins>АВ тестирование</ins>
Выполнено А/В тестирование гипотез для разных наборов данных с определением наиболее подходящего критерия проверки.
Использованные библиотеки: numpy, scipy.stats, matplotlib.pyplot, seaborn.  

Задание выполнено в [файле](training_projects/HW_AB.ipynb).

#### <ins>Проверка гипотез</ins>
В задании выполнена проверка нескольких гипотез, выбраны походящие критерии проверки в зависимости от конкретной задачи.
В ходе выполнения задания использовались библиотеки:numpy, scipy.stats, pandas, seaborn, matplotlib.pyplot.  

Задание выполнено в [файле](training_projects/ДЗ_Проверка_гипотез.ipynb).

#### <ins>Анализ данных, поиск выбросов и обработка экстремальных значений</ins>
Решение здачи классификации типа стекла в зависимости от его химического состава.   
Проведен предварительный анализ данных, определены и обработаны выбросы различными методами, выполнена классификация типов стекол и посчитаны метрики. Выполнено сравнение полученных метрик и анализ наиболее подходящего метода для обработки выбросов. 
В ходе выполнения задания спользовались библиотеки: pandas, sklearn, sklearn.RandomForestClassifier, sklearn.IsolationForest.  

Задание выполнено в [файле](training_projects/ДЗ_8_10_выбросы.ipynb).
