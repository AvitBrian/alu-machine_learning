-- lists all genres from hbtn_0d_tvshows_rate by their rating.
SELECT tv_genres.name AS name,
  IFNULL(SUM(tv_show_ratings.rate), 0) AS rating
FROM tv_genres
  LEFT JOIN tv_show_ratings ON tv_genres.id = tv_show_ratings.show_id
GROUP BY tv_genres.name
ORDER BY rating DESC;