BEGIN TRANSACTION;
CREATE TABLE Cruise_Ships(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    capacity INTEGER
);
INSERT INTO "Cruise_Ships" VALUES(1,'Grand Voyager',2000);
INSERT INTO "Cruise_Ships" VALUES(2,'Ocean Explorer',3500);
INSERT INTO "Cruise_Ships" VALUES(3,'Atlantic Adventurer',2800);
INSERT INTO "Cruise_Ships" VALUES(4,'Caribbean Princess',3600);
INSERT INTO "Cruise_Ships" VALUES(5,'Aurora Borealis',4000);
CREATE TABLE Cruise_Ships_Facilities(
    cruiseShipId INTEGER,
    facilityId INTEGER,
    FOREIGN KEY (cruiseShipId) REFERENCES Cruise_Ships(id),
    FOREIGN KEY (facilityId) REFERENCES Facilities(id),
    PRIMARY KEY (cruiseShipId, facilityId)
);
INSERT INTO "Cruise_Ships_Facilities" VALUES(1,1);
INSERT INTO "Cruise_Ships_Facilities" VALUES(2,1);
INSERT INTO "Cruise_Ships_Facilities" VALUES(3,1);
INSERT INTO "Cruise_Ships_Facilities" VALUES(4,1);
INSERT INTO "Cruise_Ships_Facilities" VALUES(5,1);
INSERT INTO "Cruise_Ships_Facilities" VALUES(5,4);
CREATE TABLE Facilities(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    facilityName TEXT,
    isAvailable BOOLEAN DEFAULT FALSE
);
INSERT INTO "Facilities" VALUES(1,'Dining Hall',1);
INSERT INTO "Facilities" VALUES(2,'Swimming Pool',0);
INSERT INTO "Facilities" VALUES(3,'Gymnasium',1);
INSERT INTO "Facilities" VALUES(4,'Theatre',0);
INSERT INTO "Facilities" VALUES(5,'Arcade',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Cruise_Ships',5);
INSERT INTO "sqlite_sequence" VALUES('Facilities',5);
COMMIT;
