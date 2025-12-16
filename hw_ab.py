
## Домашнее задание к занятию "A/B-тесты"

### Описание задачи

![banner](https://storage.googleapis.com/kaggle-datasets-images/635/1204/126be74882028aac7241553cef0e27a7/dataset-original.jpg)

Покемоны - это маленькие существа, которые сражаются друг с другом на соревнованиях. Все покемоны имеют разные характеристики (сила атаки, защиты и т. д.) И относятся к одному или двум так называемым классам (вода, огонь и т. д.).
Профессор Оук является изобретателем Pokedex - портативного устройства, которое хранит информацию обо всех существующих покемонах. Как его ведущий специалист по данным, Вы только что получили от него запрос с просьбой осуществить аналитику данных на всех устройствах Pokedex.

### Описание набора данных
Профессор Оук скопировал все содержимое в память одного устройства Pokedex, в результате чего получился набор данных, с которым Вы будете работать в этой задаче. В этом файле каждая строка представляет характеристики одного покемона:

* `pid`: Numeric - ID покемона
* `HP`: Numeric - Очки здоровья
* `Attack`: Numeric - Сила обычной атаки
* `Defense`: Numeric - Сила обычной защиты
* `Sp. Atk`: Numeric - Сила специальной атаки
* `Sp. Def`: Numeric - Сила специальной защиты
* `Speed`: Numeric - Скорость движений
* `Legendary`: Boolean - «True», если покемон редкий
* `Class 1`: Categorical - Класс покемона
* `Class 2`: Categorical - Класс покемона
"""

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings
# Отключение предупреждений (warnings)
warnings.filterwarnings("ignore")

import pandas as pd

from scipy.stats import ttest_ind
from scipy.stats import f_oneway, shapiro

pokemon = pd.read_csv('https://raw.githubusercontent.com/a-milenkin/datasets_for_t-tests/main/pokemon.csv', on_bad_lines='skip')  # Откроем датасет
pokemon.head()

# Обратите внимание, что у покемона может быть один или два класса.
# Если у покемона два класса, считается, что они имеют одинаковую значимость.

pokemon.info()

pokemon.describe()

#определение необходимого количества значений в выборке
# среднее значение стандартного отклонения по всем силам атак, защиты, скорости движений
# т.к. значения отклонения для каждой характеристики примерно одинаковые
sigma = pokemon[['Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].std().mean()
sigma

alfa = 0.05 #уровень значимости
delta = 5 # допустимая погрешность измерения - приняла минимальное значение сил из таблицы describe
z = st.norm.ppf(1-alfa/2) # квантиль нормального распределения с уровнем 1-а/2
print(f'alfa = {alfa}, допустимая погрешность измерения = {delta}, za = {z:.2f}')

n_req = int(np.ceil((z*sigma/delta)**2)) # требуемое количество значений, но дальше не использовала, т.к. в выборках получалось меньше значений
n_req

"""### Задачи

<div class="alert alert-info">
<b>Задание № 1:</b>
    
Профессор Оук подозревает, что покемоны в классе `Grass` имеют более сильную обычную атаку, чем покемоны в классе `Rock`. Проверьте, прав ли он, и убедите его в своём выводе статистически.
    
    
Примечание: если есть покемоны, которые относятся к обоим классам, просто выбросьте их;
    
Вы можете предположить, что распределение обычных атак является нормальным для всех классов покемонов.

</div>
"""

#создаю копию датафрейма с покемонами класса rock, но не в классе grass
rock = pokemon.loc[((pokemon['Class 1'] == 'Rock') | (pokemon['Class 2'] == 'Rock')) & ((pokemon['Class 1'] != 'Grass') | (pokemon['Class 2'] != 'Grass'))].copy()
rock['Class'] = 'rock'
rock_d = rock[['Attack', 'Class']].copy() #cоздаю копию только с нужными столбцами

#создаю копию датафрейма с покемонами класса grass, но не в классе rock
grass = pokemon.loc[((pokemon['Class 1'] == 'Grass') | (pokemon['Class 2'] == 'Grass')) & ((pokemon['Class 1'] != 'Rock') | (pokemon['Class 2'] != 'Rock'))].copy()
grass['Class'] = 'grass'
grass_d = grass[['Attack', 'Class']].copy() #cоздаю копию только с нужными столбцами

rock_grass = pd.concat((rock_d, grass_d)) #объединяю в один датафрейм для визуализации и проверки гипотезы

rock_grass.groupby('Class').count()

#визуализация данных
plt.figure(figsize=(5,4))
sns.barplot(x = rock_grass['Class'], y = rock_grass['Attack'], ci = False)
plt.title('Сила обычной атаки по классам')
plt.xlabel('Класс покемона')
plt.ylabel('Сила обычной атаки')
plt.show()

"""Судя по графику, в силе обычных атак классов rock и grass есть разница. Проверим путем проверки гипотезы. Причем у класса rock сила атаки больше."""

# Проверим равность дисперсий в группах
stat, p = st.levene(rock_d['Attack'],grass_d['Attack'])

print(f"Статистика = {stat:.5f}, p = {p:.5f}")

if p <0.05:
    print("Отклоняем нулевую гипотезу, вероятно, дисперсия в группах раличается")
else:
    print("Не отклоняем нулевую гипотезу, вероятно, дисперсия в группах одинаковая")

"""Т.к. по условиям задачи распределение нормальное, и, дисперсии в группах разные, то можно применить двувыборочный T-тест Стьюдента с поправкой Уэлча"""

stat, p = st.ttest_ind(rock.Attack, grass.Attack, equal_var = False) #False - Поправка Уэлча
print(f'Статистика - {stat}, p - {p}')
if p < 0.05:
  print('Отклоняем нулевую гипотезу, средние, вероятно, разные')
else:
  print('Не отклоняем нулевую гипотезу, средние, вероятно, одинаковые')

print(f'Сила обычной атаки класса rock {rock_d.Attack.mean()}')
print(f'Сила обычной атаки класса grass {grass_d.Attack.mean()}')

"""Проверка гипотезы также показала, что статистически выборки rock и grass отличаются. Причем среднее значение силы атаки для класса rock больше чем у класса grass.
Вывод: Профессор Оук ошибался.

<div class="alert alert-info">
<b>Задание № 2:</b>
    
Профессор Оук уже долго не может спать по ночам, ведь его волнует вопрос, а правда ли, что покемоны в классе `Water` в среднем быстрее, чем покемоны в классе `Normal`.
    
    
Проверьте, прав ли он, и убедите его в своём выводе статистически.
    
Примечание: если есть покемоны, которые относятся к обоим классам, выбросьте их;
    
Вы можете предположить, что распределение скорости движения является нормальным для всех классов покемонов.
</div>
"""

#создаю копию датафрейма с покемонами класса rock, но не в классе Water
water = pokemon.loc[((pokemon['Class 1'] == 'Water') | (pokemon['Class 2'] == 'Water')) & ((pokemon['Class 1'] != 'Normal') | (pokemon['Class 2'] != 'Normal'))].copy()
water['Class'] = 'water'
water_d = water[['Speed', 'Class']].copy() #cоздаю копию только с нужными столбцами

#создаю копию датафрейма с покемонами класса grass, но не в классе rock
norm = pokemon.loc[((pokemon['Class 1'] == 'Normal') | (pokemon['Class 2'] == 'Normal')) & ((pokemon['Class 1'] != 'Water') | (pokemon['Class 2'] != 'Water'))].copy()
norm['Class'] = 'normal'
norm_d = norm[['Speed', 'Class']].copy() #cоздаю копию только с нужными столбцами

water_norm = pd.concat((water_d, norm_d)) #объединяю в один

water_norm.groupby('Class').count()

water_d.describe()

norm_d.describe()

#Визуализация данных
plt.figure(figsize=(5,4))
sns.boxplot(x = 'Class', y = 'Speed', data = water_norm)
sns.swarmplot(x = 'Class', y = 'Speed', data = water_norm, color = '#BFBF00')
plt.title('Скорость по классам')
plt.xlabel('Класс покемона')
plt.ylabel('Скорость')
plt.show()

"""Cудя по графику и описательной статистике, выборки разные, но нужно проверить путем проверки гипотезы."""

#проверим имеют ли выборки равную дисперсию
stat, p = st.levene(water_d['Speed'],norm_d['Speed'])

print(f"Статистика = {stat:.5f}, p = {p:.5f}")

if p <0.05:
    print("Отклоняем нулевую гипотезу, вероятно, дисперсия в группах раличается")
else:
    print("Не отклоняем нулевую гипотезу, вероятно, дисперсия в группах одинаковая")

"""Т.к. данные распределены нормально и имеют различную дисперсию, то можно применять критерий двухвыборочный критерий Стьюдента с поправкой Уэлча"""

stat, p = st.ttest_ind(water.Speed, norm.Speed, equal_var = False) #False - поправка Уэлча
print(f'Статистика - {stat}, p - {p}')
if p < 0.05:
  print('Отклоняем нулевую гипотезу, средние, вероятно, разные')
else:
  print('Не отклоняем нулевую гипотезу, средние, вероятно, одинаковые')

print(f'Средняя скорость класса Water {water_d.Speed.mean()}')
print(f'Средняя скорость класса normal {norm_d.Speed.mean()}')

"""Т.к. проверка гипотезы показала, что выборки отличаются, а средняя скорость класса Water меньше скорости класса Normal, то профессор Оук опять ошибся - покемоны класса Water в среднем медленнее покемонов класса Normal.

<div class="alert alert-info">
<b>Задание № 3:</b>
    
Профессор Оук тот еще безумец. Он изобрёл сыворотку, способную ускорить покемона. Однако мы усомнились в эффективности его вакцины. Професоор дал эту сыворотку следующим покемонам: смотри массив `treathed_pokemon`. Проверьте, работает ли вообще его сыворотка, убедите всех в своём выводе статистически.
    
    
Вы можете предположить, что распределение скорости движения является нормальным для всех классов покемонов.

</div>
"""

# Покемоны, которые принимали сыворотку увеличения скорости
treathed_pokemon = ['Mega Beedrill', 'Mega Alakazam',
                    'Deoxys Normal Forme', 'Mega Lopunny']

axel_poks = pokemon.loc[pokemon['Name'].isin(treathed_pokemon) ].copy() # выборка ускоренных покемонов

axel_poks

usual_poks = pokemon.loc[~pokemon['Name'].isin(treathed_pokemon) ].copy() # выборка без ускоренных покемонов

# визуализация данных
sns.kdeplot(axel_poks.Speed, label = 'Ускоренные покемоны', fill = True)
sns.kdeplot(usual_poks.Speed, label = 'Обычные покемоны')
plt.title('Плотность распределения скорости')
plt.legend(['Ускоренные покемоны', 'Обычные покемоны'], loc = 'best')
plt.show()

"""Судя по графику эффект от сыворотки есть. Проверим при помощи проверки гипотезы.
Т.к. данные распределены нормально, то применим одновыборочный тест Стьюдента. Сравниваем значения скорости ускоренных покемонов со средней скоростью остальных
"""

stat, p = st.ttest_1samp(a = axel_poks['Speed'], popmean = usual_poks['Speed'].mean())
print(f'Статистика = {stat:.3f}, p = {p:.6f}')

if p < 0.05:
    print('Отклоняем нулевую гипотзу, вероятно, выборки различаются')
else:
    print('Не отклоняем нулевую гипотезу, вероятно, выборки не отличаются')

"""Молодец профессор Оук, хорошую сыворотку изобрел! Хорошо бы еще было чтобы побочек от нее не было.

<div class="alert alert-info">
<b>Задание № 4:</b>
    
Профессор Оук всегда любил истории про легендарных покемонов. Однако профессор не очень уверен, что они лучше остальных покемонов. Оук предложил разобраться в этом нам. Проверьте, действительно ли сумма характеристик `HP`,`Attack`,`Defense` у легендарных покемонов выше, чем у других покемонов?

А произведение этих же параметров?

Найдите ответы на эти вопросы и убедите всех в своём выводе статистически.
   

Вы можете предположить, что распределение сум и произведений этих параметров является нормальным для всех классов покемонов.

</div>
"""

#создаю копию датафрейма с легендарными покемонами
legend = pokemon.loc[pokemon['Legendary'] == True].copy()
legend['Legendary'] = 'Legendary'
legend['Sum'] = legend[['HP', 'Attack', 'Defense']].sum(axis = 1) # Формирую столбцы с проверяемыми значениями
legend['Prod'] = legend[['HP', 'Attack', 'Defense']].prod(axis = 1)

legend[['Sum', 'Prod']].describe()

#создаю копию датафрейма с обычными покемонами
common = pokemon.loc[pokemon['Legendary'] == False].copy()
common['Legendary'] = 'Common'
common['Sum'] = common[['HP', 'Attack', 'Defense']].sum(axis = 1) # Формирую столбцы с проверяемыми значениями
common['Prod'] = common[['HP', 'Attack', 'Defense']].prod(axis = 1)

common[['Sum', 'Prod']].describe()

#объединяю в один датафрейм и убираю лишние столбцы
comm_leg = pd.concat((legend, common))
comm_leg = comm_leg[['Legendary', 'Sum', 'Prod']] # оставляю только нужные столбцы

#Визуализация данных
plt.figure(figsize=(5,4))
sns.boxplot(x = 'Legendary', y = 'Sum', data = comm_leg)
plt.title('Суммарное значение характеристик HP, Attack, Defense')
plt.xlabel('Тип покемона')
plt.ylabel('Сумма характеристик')
plt.show()

#Визуализация данных
plt.figure(figsize=(5,4))
sns.boxplot(x = 'Legendary', y = 'Prod', data = comm_leg)
plt.title('Произведение значений характеристик HP, Attack, Defense')
plt.xlabel('Тип покемона')
plt.ylabel('Сумма характеристик')
plt.show()

"""Судя по графикам и описательной статистике сумма и произведение характеристик у редких покемонов действительно больше чем у обычных.                   
Проверим это при помощи двухвыборочного t теста Стьюдента.
"""

#для суммы характеристик
stat_s, p_s = st.ttest_ind(legend.Sum, common.Sum, equal_var = False) #False - поправка Уэлча
print(f'Для суммы характеристик статистика - {stat_s}, p - {p_s}')
if p_s < 0.05:
  print('Отклоняем нулевую гипотезу, средние, вероятно, разные')
else:
  print('Не отклоняем нулевую гипотезу, средние, вероятно, одинаковые')

#для произведения характеристик
stat_p, p_p = st.ttest_ind(legend.Prod, common.Prod, equal_var = False) #False - поправка Уэлча
print(f'Для произведения характеристик статистика - {stat_p}, p - {p_p}')
if p_p < 0.05:
  print('Отклоняем нулевую гипотезу, средние, вероятно, разные')
else:
  print('Не отклоняем нулевую гипотезу, средние, вероятно, одинаковые')

print(f'Сумма характеристик HP, Attack, Defense для легендарных покемонов {legend.Sum.mean():.2f}')
print(f'Сумма характеристик HP, Attack, Defense для обычных покемонов {common.Sum.mean():.2f}')
print('\n')
print(f'Произведение характеристик HP, Attack, Defense для легендарных покемонов {legend.Prod.mean():.2f}')
print(f'Произведение характеристик HP, Attack, Defense для обычных покемонов {common.Prod.mean():.2f}')

"""Т.к. проверка гипотез показала, что выборки с зарактеристиками для редких покемонов статистически отличаются от выборок для обычных покемонов, и средние значения редких покемонов больше чем у обычных, то можно сделать вывод, что профессор может быть уверен в превосходстве релких покемонов над обычными в части сравниваемых характеристик.

<div class="alert alert-info">
<b>Задание № 5:</b>
    
Профессор Оук частенько наблюдает за боями покемонов. После очередных таких боёв Оук выделил четыре класса `best_defence_class`, которые на его взгляд одинаковы по "силе обычной защиты" `Defense`.

Проверьте, действительно ли эти классы покемонов не отличаются по уровню защиты статистически значимо? Всё та же статистика вам в помощь!
   

Вы можете предположить, что распределение параметров защитных характеристик является нормальным для всех классов покемонов.

</div>
"""

best_defence_class = ['Rock', 'Ground', 'Steel', 'Ice']
best_defence_class

pokemon.head(1)

#создаю датафрейм с покемонами из класса rock и присваиваю им общий признак
rock = pokemon.loc[
    ((pokemon['Class 1'] == 'Rock') | (pokemon['Class 2'] == 'Rock')) &
     (~(pokemon['Class 1'].isin(['Ground', 'Steel', 'Ice'])) |~(pokemon['Class 2'].isin(['Ground', 'Steel', 'Ice'])))
    ].copy()
rock['Class'] = 'rock'

ground = pokemon.loc[
    ((pokemon['Class 1'] == 'Ground') | (pokemon['Class 2'] == 'Ground')) &
     (~(pokemon['Class 1'].isin(['Rock', 'Steel', 'Ice'])) |~(pokemon['Class 2'].isin(['Rock', 'Steel', 'Ice'])))
    ].copy()
ground['Class'] = 'ground'

steel = pokemon.loc[
    ((pokemon['Class 1'] == 'Steel') | (pokemon['Class 2'] == 'Steel')) &
     (~(pokemon['Class 1'].isin(['Rock', 'Ground', 'Ice'])) |~(pokemon['Class 2'].isin(['Rock', 'Ground', 'Ice'])))
    ].copy()
steel['Class'] = 'steel'

ice = pokemon.loc[
    ((pokemon['Class 1'] == 'Ice') | (pokemon['Class 2'] == 'Ice')) &
     (~(pokemon['Class 1'].isin(['Rock', 'Ground', 'Steel'])) |~(pokemon['Class 2'].isin(['Rock', 'Ground', 'Steel'])))
    ].copy()
ice['Class'] = 'ice'

#визуализация данных
sns.kdeplot(rock.Defense, color = '#696969')
sns.kdeplot(ground.Defense, color = '#D2691E')
sns.kdeplot(steel.Defense, color = '#6495ED')
sns.kdeplot(ice.Defense, color = '#00FFFF')
plt.title('Плотность вероятности обычной защиты по классам')
plt.legend(['rock', 'ground', 'steel', 'ice'], loc = 'best')
plt.show()

"""В данном случае по графику сложно сказать что-то определенное.       
Статистическую равность или разность выборок проверим при помощи однофакторного дисперсионного анализа, т.к. выборок 4 шт.
Перед этим проверим равность дисперсий
"""

stat, p = st.levene(rock.Defense, ground.Defense, steel.Defense, ice.Defense)
print(f'Статистика {stat}, p {p}')
if p < 0.05:
  print(f'Отклоняем нулевую гипотезу, вероятно, вариации разные')
else:
  print(f'Принимаем нулевую гипотезу, вероятно, вариации равны')

fvalue, pvalue = st.f_oneway(rock.Defense, ground.Defense, steel.Defense, ice.Defense)
print(f'Статистика f-value {fvalue:.5f}, статистика p-value {fvalue:.5f}')
if pvalue < 0.05:
  print('Отклоняем нулевую гипотезу, вероятно средние различаются')
else:
  print('Принимаем нулевую гипотезу, вероятно средние одинаковые')

"""Тест показал, что какие то из выборок различаются.      
Проверим при помощи критерия Тьюки
"""

#объединяю в один датафрейм для передачи в метод криерия Тьюки, оставляю только информативные столбцы
four_groups = pd.concat((rock[['Defense', 'Class']], ground[['Defense', 'Class']], steel[['Defense', 'Class']], ice[['Defense', 'Class']]))

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog = four_groups['Defense'].values,
                          groups = four_groups['Class'],
                          alpha=0.05)
print(tukey)

tukey.plot_simultaneous(comparison_name='rock')
tukey.plot_simultaneous(comparison_name='ground')
plt.show()

"""Проверка гипотеза показала, что профессор Оук несколько ошибался насчет равности всех 4-х классов между собой по силе прямой защиты.      
Статистически не различаются между собой только классы ice, ground и steel, rock.
"""