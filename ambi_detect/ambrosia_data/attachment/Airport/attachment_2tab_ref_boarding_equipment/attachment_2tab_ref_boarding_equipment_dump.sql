BEGIN TRANSACTION;
CREATE TABLE `Aircraft` (
    `Id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `Type` TEXT,
    `Seats` INT);
INSERT INTO "Aircraft" VALUES(1,'Boing 737',200);
INSERT INTO "Aircraft" VALUES(2,'Airbus A380',500);
INSERT INTO "Aircraft" VALUES(3,'Fokker F28',60);
CREATE TABLE `Flight` (
    `Id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `Aircraft_id` INTEGER,
    `Terminal_id` INTEGER,
    `Departure_time` DATETIME,
    `Arrival_time` DATETIME,
    FOREIGN KEY(Aircraft_id) REFERENCES `Aircraft`(`Id`),
    FOREIGN KEY(Terminal_id) REFERENCES `Terminals`(`Id`));
INSERT INTO "Flight" VALUES(1,1,1,'2022-12-01 09:00:00','2022-12-01 11:00:00');
INSERT INTO "Flight" VALUES(2,2,1,'2022-12-02 10:00:00','2022-12-02 12:00:00');
INSERT INTO "Flight" VALUES(3,3,2,'2022-12-01 14:00:00','2022-12-01 16:00:00');
INSERT INTO "Flight" VALUES(4,4,2,'2022-12-02 15:00:00','2022-12-02 17:00:00');
INSERT INTO "Flight" VALUES(5,5,1,'2022-12-01 18:00:00','2022-12-01 20:00:00');
CREATE TABLE `Jet_bridge` (
    `Id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `Length` REAL,
    `Capacity` INT,
    `Terminal_id` INTEGER,
    FOREIGN KEY(Terminal_id) REFERENCES `Terminals`(`Id`));
INSERT INTO "Jet_bridge" VALUES(1,90.0,100,1);
INSERT INTO "Jet_bridge" VALUES(2,100.0,120,2);
CREATE TABLE `Staircar` (
    `Id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `Number_of_Cars` INT,
    `Capacity` INT,
    `Terminal_id` INTEGER,
    FOREIGN KEY(Terminal_id) REFERENCES `Terminals`(`Id`));
INSERT INTO "Staircar" VALUES(1,10,50,1);
INSERT INTO "Staircar" VALUES(2,5,30,2);
CREATE TABLE `Terminals` (
    `Id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `Name` TEXT UNIQUE);
INSERT INTO "Terminals" VALUES(1,'Terminal 1');
INSERT INTO "Terminals" VALUES(2,'Terminal 2');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Terminals',2);
INSERT INTO "sqlite_sequence" VALUES('Aircraft',3);
INSERT INTO "sqlite_sequence" VALUES('Jet_bridge',2);
INSERT INTO "sqlite_sequence" VALUES('Staircar',2);
INSERT INTO "sqlite_sequence" VALUES('Flight',5);
COMMIT;
