import pandas as pd
import sys
import re
import os
import glob
from datetime import datetime
import time

def act_analys(deal_list):
  """Функция выполняет разбор и учет дейсвий игроков по правилам.
  На вход подается подготовленный список игроков с действиями ОДНОЙ раздачи.
  На выход датафрейм, в который заносятся результаты одной раздачи play_1_hand.
  Заходит список игроков и действий префлопа deal_list
  """
  play_1_hand = pd.DataFrame(columns = ['player_id', 'PFR',	'VPIP',	'o_Limp',	'3bet', 'VPIP/PFR']) #создаю промежуточный датафрейм для каждой раздачи по типу итогового

  #start_time = datetime.now() #засекаю время начала работы
  #print(f'на вход приходит: {deal_list}')
  call_flag = 0
  rais_flag = 0
  for line in deal_list: # пробегаем по всему списку игроков
    #print(f'Начинаем разбирать {line}')
    s_line = line.split(':') # делим строку по :. s_line[0] - имя игрока, s_line[1] - действие
    #print(f'Расчленение строки по : заняло {datetime.now() - start_time}')
    #start_time = datetime.now()

    if any(act in s_line[1] for act in ['call', 'rais', 'check', 'fold']): # проверяю чтобы в строке было нужное действие
      #print(f'Проверка наличия ключевых слов заняло {datetime.now() - start_time}')
      #start_time = datetime.now()

      pfr = 0 # исходное значение параметра
      vpip = 0
      o_Limp = 0
      three_bet = 0

      if ('call' in s_line[1]) and (call_flag == 1):
        #print('Случай 1')
        vpip = 1
        #print(f'vpip = {vpip}')
      elif ('call' in s_line[1]) and (call_flag == 0) and (rais_flag == 0):
        #print('Случай 2')
        vpip, o_Limp = 1, 1
        call_flag = 1 #поднимаем флаг, что call уже был
        #print(f'vpip = {vpip}, o_Limp = {o_Limp}, call_flag = {call_flag}')
      if ('call' in s_line[1]) and (call_flag == 0):
        vpip = 1
        call_flag = 1
      elif ('rais' in s_line[1]) and (rais_flag == 1):
        #print('Случай 3')
        pfr, vpip, three_bet = 1, 1, 1
        rais_flag += 1
        #print(f'vpip = {vpip}, pfr = {pfr}, three_bet = {three_bet}, rais_flag = {rais_flag}')
      elif ('rais' in s_line[1]) and (rais_flag == 0): # если rais и до этого не было rais, то pfr=1, vpip=1
        #print('Случай 4')
        pfr, vpip = 1, 1
        rais_flag += 1
        #print(f'vpip = {vpip}, pfr = {pfr}, rais_flag = {rais_flag}')
      elif ('rais' in s_line[1]) and (rais_flag > 1 ):
        #print('Случай 5')
        pfr, vpip = 1, 1
        rais_flag += 1

      #print(f'Проверка правил заняла {datetime.now() - start_time}')
      #start_time = datetime.now()

      if ~play_1_hand['player_id'].isin([s_line[0]]).any(): # если игрок еще не внесен в базу
        play_1_hand.loc[len(play_1_hand)] = [s_line[0], pfr, vpip, o_Limp, three_bet, 0] #добавляю новую строку с именем игрока и вычисленными действиями

      else: #если игрок уже был в базе, то проверяем дублируются действие или нет. Если нет - добавляем
        if (play_1_hand.loc[play_1_hand['player_id'] == s_line[0]]['PFR'].item() == 0) and ( pfr == 1 ): #если для этого игрока еще не было pfr, а мы его нашли
          play_1_hand.loc[ play_1_hand['player_id'] == s_line[0], 'PFR' ] = pfr # то присваеваем pfr игрока найденному
          """ в других случаях уже ничего не делаем.
          если pfr игрока уже 1, то не надо его считать 2-й раз.
          а если найденный pfr =0, то и нечего увеличивать
          """
        if (play_1_hand.loc[play_1_hand['player_id'] == s_line[0]]['VPIP'].item() == 0) and ( vpip == 1 ): # подход по vpip аналогчен подходу по pfr
          play_1_hand.loc[ play_1_hand['player_id'] == s_line[0], 'VPIP' ] = vpip

        if (play_1_hand.loc[play_1_hand['player_id'] == s_line[0]]['o_Limp'].item() == 0) and ( o_Limp == 1 ):
          play_1_hand.loc[ play_1_hand['player_id'] == s_line[0], 'o_Limp' ] = o_Limp

        if (play_1_hand.loc[play_1_hand['player_id'] == s_line[0]]['3bet'].item() == 0) and ( three_bet == 1 ):
          play_1_hand.loc[ play_1_hand['player_id'] == s_line[0], '3bet' ] = three_bet



  return play_1_hand

direct = r'C:\Mary\work_var\files' #текстовые файлы с раздачами

start_time = datetime.now() #засекаю время начала работы
for path in glob.iglob(f'{direct}/**/*.txt', recursive = True):
  with open(path, 'rt') as f: #открываю каждый файл для чтения
    #print(f'Открытие файла {path} заняло {datetime.now() - start_time}')
    print(f'Читаю файл {path}')
    #start_time = datetime.now()

    try:
        deals_t = f.read() #считываю содержимое текущего файла в текстовую переменную deals_t
        #print(f'Чтение файла {path} заняло {datetime.now() - start_time}')
        #start_time = datetime.now()

        #print(f'длина файла {file} - {len(deals_t)}') #проверка корректности считывания
        act_list = re.findall(r'HOLE CARDS \*\*\*[\w\W\s]*?\*\*\* ', deals_t) # выбираю из считанного текста нужные куски текста по шаблону между HOLE CARDS и следующими *
        #print(f'Выборка нужных кусков из файла {path} заняло {datetime.now() - start_time}')
        #start_time = datetime.now()

        deals = [] #создаю пустой список с очищенными списками раздач, который потом передается для обработки действий

        players = pd.DataFrame(columns = ['player_id', 'PFR',	'VPIP',	'o_Limp',	'3bet', 'VPIP/PFR']) #создаю пустой ИТОГОВЫЙ датафрейм с нужными столбцами, в который буду записывать игроков и их действия

        # формирую deals список, содержащий списки каждой раздачи
        for l in act_list: #для каждого элемента списка (одна раздача)
          split_list = re.split('\n', l) #разделяю строку с одной раздачей на подстроки каждого игрока, разделитель перенос строки
          #clear_list очищенный список игроков и их действий одной раздачи
          clear_list = [p for p in split_list if ':' in p] # оставляю из списка только подстроки действий игроков
          deals.append(clear_list) #добавляю в общий список раздач очищенный список одной раздачи
        #print(f'Формирование списка списков раздач заняло {datetime.now() - start_time}')
        #start_time = datetime.now()

        for deal_list in deals:
          #print(deal_list)
          play_1_hand = act_analys(deal_list)
          players = pd.concat([players, play_1_hand], axis = 0) #Присоединяю по строкам датафрейм одной раздачи к итоговому датафорейму

        #print(f'Разбор раздач по правилам и запись в ДФ заняли {datetime.now() - start_time}')
        #start_time = datetime.now()

        #Записать полученный датафрейм в файл csv
        #индексы из датафрейма мне не нужны, заголовки тоже, т.к. буду дозаписывать файл
        #заголовок добавлю когда буду считывать полный датафрейм для кластеризации
        #файл открыт в режиме добавления - mode = a

        players.to_csv(r'C:\Mary\work_var\players.csv', mode = 'a', index = False, header = False) # ссылка дома
    except:
      pass
print(f'Обработка заняла {datetime.now() - start_time} mc')

#вес полученного списка списков, байты
sys.getsizeof(deals)

# вес сформированного датафрейма байты
sys.getsizeof(players)
