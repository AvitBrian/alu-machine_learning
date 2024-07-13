-- Creates a trigger that resets the validation of an email after it is updated
DROP TRIGGER IF EXISTS reset_validation;
DELIMITER $$ CREATE TRIGGER reset_validation BEFORE
UPDATE ON users FOR EACH ROW BEGIN IF NEW.email <> OLD.email THEN
SET NEW.valid_email = 0;
END IF;
END $$ DELIMITER;