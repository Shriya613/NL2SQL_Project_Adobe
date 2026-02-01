BEGIN TRANSACTION;
CREATE TABLE Bar(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    DrinkSelection TEXT,
    OpeningTime TEXT);
INSERT INTO "Bar" VALUES(1,'Tap House','123 Vineyard Lane','Wines & Beers','5 p.m. - midnight.');
INSERT INTO "Bar" VALUES(2,'Martini Lounge','456 Olive Street','Cocktails','4 p.m. - 2 a.m.');
CREATE TABLE Buffet(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    Capacity INTEGER,
    Hours TEXT);
INSERT INTO "Buffet" VALUES(1,'Grand Brunch','123 Main St',100,'7.30 a.m. - 4 p.m.');
INSERT INTO "Buffet" VALUES(2,'The All Day Diner','456 Elm St',80,'9 a.m. - 11 p.m.');
CREATE TABLE Coffee_Shop(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    SeatingCapacity INTEGER,
    Hours TEXT);
INSERT INTO "Coffee_Shop" VALUES(1,'Bean There Done That','789 Oak St',50,'7.30 a.m. - 4 p.m.');
INSERT INTO "Coffee_Shop" VALUES(2,'Brewed Awakenings','135 Pine St',60,'9 a.m. - 6 p.m.');
CREATE TABLE Event(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Venue TEXT,
    Date TEXT,
    Time TEXT);
INSERT INTO "Event" VALUES(1,'Annual Gala','Convention Center','Dec 31st','8 p.m. - 12 a.m.');
INSERT INTO "Event" VALUES(2,'Summer Jazz Festival','Park Amphitheater','July 4th','6 p.m. - 10 p.m.');
CREATE TABLE Restaurant(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    Cuisine TEXT,
    OperatingTime TEXT);
INSERT INTO "Restaurant" VALUES(1,'Le Petit Bistro','246 Maple Ave','French','Noon - 10 p.m.');
INSERT INTO "Restaurant" VALUES(2,'Mediterranean Delights','321 Birch Blvd','Mediterranean','11 a.m. - Midnight');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Buffet',2);
INSERT INTO "sqlite_sequence" VALUES('Coffee_Shop',2);
INSERT INTO "sqlite_sequence" VALUES('Restaurant',2);
INSERT INTO "sqlite_sequence" VALUES('Bar',2);
INSERT INTO "sqlite_sequence" VALUES('Event',2);
COMMIT;
