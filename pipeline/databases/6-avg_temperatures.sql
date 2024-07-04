-- displays the average temperature (Fahrenheit) by city ordered by temperature (descending)
USE hbtn_0c_0
source temperatures.sql
SELECT city, AVG("value") AS avg_temp
FROM table_name
GROUP BY city
ORDER BY avg_temp DESC;