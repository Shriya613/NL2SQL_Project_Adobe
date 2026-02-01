BEGIN TRANSACTION;
CREATE TABLE `airports`(
    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `name` TEXT UNIQUE);
INSERT INTO "airports" VALUES(1,'Auckland');
INSERT INTO "airports" VALUES(2,'Los Angeles');
INSERT INTO "airports" VALUES(3,'Tokyo');
INSERT INTO "airports" VALUES(4,'Dubai');
INSERT INTO "airports" VALUES(5,'New York');
CREATE TABLE `flights`(
    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `from_airport_id` INTEGER,
    `to_airport_id` INTEGER,
    FOREIGN KEY(`from_airport_id`) REFERENCES `airports`(`id`),
    FOREIGN KEY(`to_airport_id`) REFERENCES `airports`(`id`));
INSERT INTO "flights" VALUES(1,1,2);
INSERT INTO "flights" VALUES(2,3,4);
INSERT INTO "flights" VALUES(3,5,6);
INSERT INTO "flights" VALUES(4,7,8);
INSERT INTO "flights" VALUES(5,9,10);
CREATE TABLE `manufacturers`(
    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `name` TEXT UNIQUE);
INSERT INTO "manufacturers" VALUES(1,'Boeing');
INSERT INTO "manufacturers" VALUES(2,'Airbus');
INSERT INTO "manufacturers" VALUES(3,'Embraer');
INSERT INTO "manufacturers" VALUES(4,'De Havilland Canada');
CREATE TABLE `plane_details`(
    `plane_id` INTEGER,
    `manufacturer_id` INTEGER,
    FOREIGN KEY(`plane_id`) REFERENCES `planes`(`id`),
    FOREIGN KEY(`manufacturer_id`) REFERENCES `manufacturers`(`id`));
INSERT INTO "plane_details" VALUES(1,1);
INSERT INTO "plane_details" VALUES(2,1);
INSERT INTO "plane_details" VALUES(3,2);
INSERT INTO "plane_details" VALUES(4,3);
INSERT INTO "plane_details" VALUES(5,4);
CREATE TABLE "planes"(
    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `model` TEXT,
    `number_of_seats` INTEGER);
INSERT INTO "planes" VALUES(1,'Boeing 787',180);
INSERT INTO "planes" VALUES(2,'Boeing 787',200);
INSERT INTO "planes" VALUES(3,'Airbus A380',180);
INSERT INTO "planes" VALUES(4,'Airbus A380',200);
INSERT INTO "planes" VALUES(5,'Bombardier CRJ900',88);
INSERT INTO "planes" VALUES(6,'Bombardier Q400',78);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('airports',5);
INSERT INTO "sqlite_sequence" VALUES('flights',5);
INSERT INTO "sqlite_sequence" VALUES('manufacturers',4);
INSERT INTO "sqlite_sequence" VALUES('planes',6);
COMMIT;
