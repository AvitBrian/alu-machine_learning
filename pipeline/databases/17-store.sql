-- Creates a trigger that decreases the quantity of an item after an order is made
DELIMITER $$ CREATE TRIGGER add_order
AFTER
INSERT ON orders FOR EACH ROW BEGIN
UPDATE items
SET quantity = quantity - NEW.number
WHERE items.name = NEW.item_name;
END $$ DELIMITER;