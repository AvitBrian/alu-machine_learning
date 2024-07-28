-- resets the validation of an email after it is updated
DELIMITER | CREATE TRIGGER TRIGGER_Products_Insert
AFTER
INSERT ON Products FOR EACH ROW BEGIN
UPDATE Products
SET current = 0
WHERE id = new.id
	AND current = 1
	AND autonumber <> new.autonumber;
END;
| DELIMITER;