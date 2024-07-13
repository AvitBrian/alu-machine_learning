-- ranks the country origins of bands, ordered by the number of (non-unique) fans
SELECT origin,
  SUM(fans) AS nb_fans
FROM metal_bands
WHERE NOT DISTINCT fans
GROUP BY origin
ORDER BY nb_fans DESC;