-- displays the max temperature by state ordered by state
SELECT state, MAX(value) AS max_temp
FROM temperatures
GROUP BY max_temp
ORDER BY state;