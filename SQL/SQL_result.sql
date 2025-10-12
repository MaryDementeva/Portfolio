set search_path to public;

/*
 1. Выведите название самолетов, которые имеют менее 50 посадочных мест?
 */
select a_q.model, a_q.quont_seats
from (
	select a.model, count(*) as quont_seats
	from aircrafts a
	join seats s on a.aircraft_code = s.aircraft_code 
	group by a.aircraft_code) a_q
where a_q.quont_seats <50

--2. Выведите процентное изменение ежемесячной суммы бронирования билетов, округленной до сотых.
select t."Year", t."Month", t.sum as "Summary booking",
		case
			when lag(t.sum) over (order by t."Year", t."Month") = 0.0 then null
			else round((t.sum - lag(t.sum) over (order by t."Year", t."Month"))*100/lag(t.sum) over (order by t."Year", t."Month"),2)
		end as "Monthly change,%"
from (
	select date_part('Year', book_date) as "Year", date_part('Month', book_date) as "Month", sum(total_amount)
	from bookings
	group by date_part('Year', book_date),  date_part('Month', book_date)) t


--3. Выведите названия самолетов не имеющих бизнес - класс. Решение должно быть через функцию array_agg.
select t.aircraft_code, t.model
from (
	select a.aircraft_code, a.model, array_agg (distinct lower(s.fare_conditions))
	from aircrafts a 
	join seats s on a.aircraft_code = s.aircraft_code 
	group by a.aircraft_code) t
where array_position (t.array_agg, lower('Business')) is null

/*
 5. Найдите процентное соотношение перелетов по маршрутам от общего количества перелетов.
 Выведите в результат названия аэропортов и процентное отношение.
 Решение должно быть через оконную функцию.
 */

select a.airport_name as departure_airport, 
	a1.airport_name as arrival_airport, 
	round(count(f.flight_no)*100/sum(count(f.flight_id)) over (),2) as "route/flights,%"
from flights f
left join airports a on f.departure_airport = a.airport_code 
left join airports a1 on f.arrival_airport = a1.airport_code 
group by f.flight_no, a.airport_name, a1.airport_name 

/*проверка. в сумме 100%. Из-за округления до 2-го знака пропадает около 3-х %
select sum(t."route/flights,%") 
from (
	select a.airport_name as departure_airport, 
	a1.airport_name as arrival_airport, 
	count(f.flight_no)*100/sum(count(f.flight_id)) over () as "route/flights,%"
from flights f
left join airports a on f.departure_airport = a.airport_code 
left join airports a1 on f.arrival_airport = a1.airport_code 
group by f.flight_no, a.airport_name, a1.airport_name) t */  



--6. Выведите количество пассажиров по каждому коду сотового оператора, если учесть, что код оператора - это три символа после +7


/*select *
from tickets t 

select contact_data, pg_typeof(contact_data) --jsonb, проверяю какой ип данных у столбца contact_data
from tickets

select jsonb_object_keys(contact_data) --phone, проверяю какие ключи есть и какой нужен мне для поиска
from tickets

select contact_data->'phone', pg_typeof(contact_data->'phone')--проверяю в каком типе данные по ключу phone - jsonb
from tickets

select contact_data->>'phone', pg_typeof(contact_data->>'phone')--перевожу данные по ключу phone в текст, проверяю тип переведенных данных
from tickets */

--explain analyze --cost 83372, time 1526
select sub.phone_code, count(sub.phone_code) as num_of_clients
from (
	select (substr(contact_data->>'phone', strpos(contact_data->>'phone','+7')+2, 3)) as phone_code --cost 
	from tickets
	where strpos(contact_data->>'phone','+7') <> 0) sub --scan cost 10718.6, time 294.3, sort cost 300, time 100 
group by sub.phone_code --cost 6500, time 150, cost 41000, time 200 (gather merge)

/*7. Классифицируйте финансовые обороты (сумма стоимости перелетов) по маршрутам:
 До 50 млн - low
 От 50 млн включительно до 150 млн - middle
 От 150 млн включительно - high
 Выведите в результат количество маршрутов в каждом полученном классе */

--explain analyze --cost 18805, time 909ms
select c.classification, count(*) as count_flights
from(
	select f.flight_no,
		sum(tf.amount),
		case 
			when sum(tf.amount) < 50000000 then 'low'
			when sum(tf.amount) >= 50000000 and sum(tf.amount) < 150000000 then 'middle'
			else 'high'
		end as classification,
		count(*) as count_of_flights
	from flights f 
	join ticket_flights tf on f.flight_id  = tf.flight_id 
	group by f.flight_no) c
group by c.classification


/* 8. Вычислите медиану стоимости перелетов, 
 * медиану размера бронирования 
 * и отношение медианы бронирования к медиане стоимости перелетов, округленной до сотых*/

--медиана стоимости перелетов
/*select percentile_cont(0.5) within group (order by amount)
from ticket_flights tf 

--медиана стоимости бронирования
select percentile_cont(0.5) within group (order by total_amount) 
from bookings
*/
explain analyze --cost 30016, time 1998ms
with median_flights as (
	select percentile_cont(0.5) within group (order by amount)
	from ticket_flights) 
select percentile_cont(0.5) within group (order by total_amount) as "Медиана стоимости бронирования",
	(select * from median_flights) as "Медиана стоимости перелетов",
	round ((percentile_cont(0.5) within group (order by total_amount)/(select * from median_flights))::numeric,2) as "Медиана брони/медиана перелетов"
from bookings
	

/* 9. Найдите значение минимальной стоимости полета 1 км для пассажиров. 
 * То есть нужно найти расстояние между аэропортами и с учетом стоимости перелетов получить искомый результат
 */

create extension cube

create extension earthdistance 


--drop extension earthdistance cascade

--drop extension cube


select min(unit_cost) as min_unit_cost_per_km  
from (
	select min((tf.amount/(earth_distance(ll_to_earth(a.latitude, a.longitude), ll_to_earth(a1.latitude, a1.longitude))/1000))) as unit_cost
	from flights f
	left join ticket_flights tf on f.flight_id  = tf.flight_id 
	left join airports a on f.departure_airport = a.airport_code 
	left join airports a1 on f.arrival_airport = a1.airport_code) t





