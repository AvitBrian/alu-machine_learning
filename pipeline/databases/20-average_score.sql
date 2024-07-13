-- creates a stored procedure ComputeAverageScoreForUser that computes the average score of a user and updates the average_score field in the users table.
DELIMITER // CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id_new INTEGER) BEGIN
UPDATE users
SET average_score =(
		SELECT AVG(score)
		FROM corrections
		WHERE user_id = user_id_new
	)
WHERE id = user_id_new;
END;
// DELIMITER;