--создаю схему, в которой хранится таблица
create schema VK;

set search_path to VK; --указываю путь к новой схеме

-- СОЗДАНИЕ ТАБЛИЦЫ--
create table VK_likes (
	post_date timestamp,
	num_likes int not null 
	)
	
--drop table VK_likes cascade
	
--ВНЕСЕНИЕ ДАННЫХ--
insert into VK_likes
values ('2021-09-18 21:24:00',13),
		('2021-07-03 12:17:00',14),
		('2021-05-16 12:53:00',15),
		('2021-04-10 11:44:00',9),
		('2021-04-04 11:52:00',12),
		('2021-04-04 11:52:00',11),
		('2021-03-28 21:59:00',11),
		('2021-03-08 22:19:00',22),
		('2021-03-08 22:18:00',19),
		('2021-03-08 22:16:00',23),
		('2021-02-23 22:54:00',12),
		('2021-01-13 22:08:00',31),
		('2021-01-13 21:51:00',3),
		('2021-01-10 21:48:00',6),
		('2020-11-17 12:30:00',0),
		('2020-11-10 18:48:00',26),
		('2020-11-10 18:43:00',10),
		('2020-11-08 12:12:00',7),
		('2020-09-10 19:01:00',8),
		('2020-08-18 12:10:00',5),
		('2020-08-10 17:25:00',5),
		('2020-07-01 17:14:00',6),
		('2020-06-24 13:44:00',18),
		('2020-04-03 17:44:00',10),
		('2020-02-18 08:44:00',0),
		('2018-07-30 22:19:00',7),
		('2018-05-20 10:55:00',33),
		('2012-02-23 22:26:00',4)
		
		
select * from VK_likes

--1. Корреляция времени суток и количества лайков--
with hod as(
   select extract(hour from post_date) as hour_of_day,
   		  num_likes as likes
   from VK_likes
  ),
result as (  
	select 
		case when (hour_of_day >= 0) and (hour_of_day <= 4) then 1 -- ночь
	     	when (hour_of_day >= 5) and (hour_of_day <= 11) then 2 -- утро
		 	when (hour_of_day >= 12) and (hour_of_day <= 17) then 3 -- день
		else 4 -- вечер
		end as part_of_day,
		likes
	from hod
  )
 select corr(part_of_day, likes) as corr_hour
 from result
 
 --получили корреляцию между временем суток и количеством лайков 0,115
 
 --2. Корреляция дня недели и количества лайков--
 
with dow as(
   select extract(dow from post_date) as day_of_week,
   		  num_likes as likes
   from VK_likes
  )
 select corr(day_of_week, likes) as day_corr
 from dow
 
 --получили корреляцию -0,12--
 
 --3. корреляция промежутка между постами--
 --неверный вариант - не надо суммировать лайки по границам интервала
 --with res_interval as(
	--select 
		--post_date - LAG(post_date, 1, null) over (order by post_date) as interval,
		--num_likes + LAG(num_likes, 1, 0) over (order by post_date) as sum_slots
	--from VK_likes
--)
--select corr(extract(epoch from interval), sum_slots) as interval_corr
--from res_interval

--получили корреляцию 0,133--




with res_interval as(
	select 
		post_date - LAG(post_date, 1, null) over (order by post_date) as interval,
		num_likes as likes
	from VK_likes
)
select corr(extract(epoch from interval), likes) as interval_corr
from res_interval