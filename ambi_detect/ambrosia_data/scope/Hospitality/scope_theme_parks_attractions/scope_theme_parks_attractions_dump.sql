BEGIN TRANSACTION;
CREATE TABLE Attractions(
    attraction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    description TEXT);
INSERT INTO "Attractions" VALUES(1,'Roller Coaster','A thrilling high-speed ride.');
INSERT INTO "Attractions" VALUES(2,'Ferris Wheel','An observation wheel that gives riders a panoramic view of the surroundings.');
INSERT INTO "Attractions" VALUES(3,'Bumper Cars','Cars mounted on rotating platforms so they can bump into one another.');
INSERT INTO "Attractions" VALUES(4,'Carousel','A merry-go-round featuring various animals for riders to sit on.');
INSERT INTO "Attractions" VALUES(5,'Haunted House','A themed dark ride intended to scare passengers.');
CREATE TABLE Theme_Parks(
    park_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Theme_Parks" VALUES(1,'FunLand','New York');
INSERT INTO "Theme_Parks" VALUES(2,'ThrillZone','California');
INSERT INTO "Theme_Parks" VALUES(3,'AdventureWorld','Florida');
INSERT INTO "Theme_Parks" VALUES(4,'Wonderland','Texas');
INSERT INTO "Theme_Parks" VALUES(5,'AmusementCentral','Illinois');
CREATE TABLE Theme_Parks_Attractions(
    theme_park_id INTEGER REFERENCES Theme_Parks(park_id),
    attraction_id INTEGER REFERENCES Attractions(attraction_id));
INSERT INTO "Theme_Parks_Attractions" VALUES(1,1);
INSERT INTO "Theme_Parks_Attractions" VALUES(2,1);
INSERT INTO "Theme_Parks_Attractions" VALUES(3,1);
INSERT INTO "Theme_Parks_Attractions" VALUES(4,1);
INSERT INTO "Theme_Parks_Attractions" VALUES(5,1);
INSERT INTO "Theme_Parks_Attractions" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Theme_Parks',5);
INSERT INTO "sqlite_sequence" VALUES('Attractions',5);
COMMIT;
