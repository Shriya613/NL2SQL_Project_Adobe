BEGIN TRANSACTION;
CREATE TABLE Amusement_Parks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Amusement_Parks" VALUES(1,'Disneyland','California');
INSERT INTO "Amusement_Parks" VALUES(2,'Universal Studios','Florida');
INSERT INTO "Amusement_Parks" VALUES(3,'SeaWorld','Texas');
INSERT INTO "Amusement_Parks" VALUES(4,'Six Flags Magic Mountain','California');
INSERT INTO "Amusement_Parks" VALUES(5,'Legoland','California');
CREATE TABLE Amusement_Parks_Rides(
    amusement_park_id INTEGER,
    rides_id INTEGER,
    FOREIGN KEY(amusement_park_id) REFERENCES Amusement_Parks(id),
    FOREIGN KEY(rides_id) REFERENCES Rides(id));
INSERT INTO "Amusement_Parks_Rides" VALUES(1,1);
INSERT INTO "Amusement_Parks_Rides" VALUES(2,1);
INSERT INTO "Amusement_Parks_Rides" VALUES(3,1);
INSERT INTO "Amusement_Parks_Rides" VALUES(4,1);
INSERT INTO "Amusement_Parks_Rides" VALUES(5,1);
INSERT INTO "Amusement_Parks_Rides" VALUES(5,4);
CREATE TABLE Rides(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ride_name TEXT,
    ride_type TEXT);
INSERT INTO "Rides" VALUES(1,'Ferris Wheel','Attraction');
INSERT INTO "Rides" VALUES(2,'Roller Coaster','Thrill Ride');
INSERT INTO "Rides" VALUES(3,'Carousel','Family Ride');
INSERT INTO "Rides" VALUES(4,'Bumper Cars','Kids Ride');
INSERT INTO "Rides" VALUES(5,'Water Rides','Wet & Wild');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Amusement_Parks',5);
INSERT INTO "sqlite_sequence" VALUES('Rides',5);
COMMIT;
