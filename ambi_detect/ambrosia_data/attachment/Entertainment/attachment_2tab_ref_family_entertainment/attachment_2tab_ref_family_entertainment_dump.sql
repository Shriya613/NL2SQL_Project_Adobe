BEGIN TRANSACTION;
CREATE TABLE AmusementPark (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    locationID INTEGER,
    FOREIGN KEY(locationID) REFERENCES Location(id));
INSERT INTO "AmusementPark" VALUES(1,'Disney World',1);
INSERT INTO "AmusementPark" VALUES(2,'Universal Studios Hollywood',2);
CREATE TABLE Location (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    state TEXT);
INSERT INTO "Location" VALUES(1,'Orlando','Florida');
INSERT INTO "Location" VALUES(2,'Los Angeles','California');
CREATE TABLE Rides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parkID INTEGER,
    rideName TEXT,
    description TEXT,
    FOREIGN KEY(parkID) REFERENCES AmusementPark(id));
INSERT INTO "Rides" VALUES(1,1,'Space Mountain','A classic roller coaster experience.');
INSERT INTO "Rides" VALUES(2,1,'It''s a Small World','An iconic boat ride through miniature scenes of children from countries around the world.');
INSERT INTO "Rides" VALUES(3,1,'Haunted Mansion','Explore the mysteries and spirits that inhabit this eerie Victorian mansion.');
INSERT INTO "Rides" VALUES(4,1,'The Twilight Zone Tower of Terror','Plunge down a vertical drop on an elevator ride like no other!');
INSERT INTO "Rides" VALUES(5,1,'Buzz Lightyear''s Space Ranger Spin','Help Buzz save the galaxy by blasting away at targets.');
INSERT INTO "Rides" VALUES(6,2,'Jurassic Park - The Ride','Experience the thrill of being chased by dinosaurs on this whitewater raft adventure.');
CREATE TABLE WaterPark (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    locationID INTEGER,
    FOREIGN KEY(locationID) REFERENCES Location(id));
INSERT INTO "WaterPark" VALUES(1,'Typhoon Lagoon',1);
INSERT INTO "WaterPark" VALUES(2,' Universal CityWalk',2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Location',2);
INSERT INTO "sqlite_sequence" VALUES('AmusementPark',2);
INSERT INTO "sqlite_sequence" VALUES('WaterPark',2);
INSERT INTO "sqlite_sequence" VALUES('Rides',6);
COMMIT;
