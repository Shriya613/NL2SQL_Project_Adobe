BEGIN TRANSACTION;
CREATE TABLE HouseFeatures(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    featureName TEXT
);
INSERT INTO "HouseFeatures" VALUES(1,'Garage');
INSERT INTO "HouseFeatures" VALUES(2,'Pool');
INSERT INTO "HouseFeatures" VALUES(3,'Balcony');
INSERT INTO "HouseFeatures" VALUES(4,'Air Conditioning');
INSERT INTO "HouseFeatures" VALUES(5,'Fireplace');
CREATE TABLE Houses(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    price REAL,
    numOfFloors INTEGER ,
    FOREIGN KEY (id) REFERENCES Houses_Rooms(houseId) ON DELETE CASCADE
);
INSERT INTO "Houses" VALUES(1,'123 Main St',300000.0,2);
INSERT INTO "Houses" VALUES(2,'456 Oak Ave',400000.0,3);
INSERT INTO "Houses" VALUES(3,'789 Pine St',500000.0,4);
INSERT INTO "Houses" VALUES(4,'101 Elm Blvd',600000.0,3);
INSERT INTO "Houses" VALUES(5,'202 Maple Dr',700000.0,2);
CREATE TABLE Houses_Rooms(
    houseId INTEGER,
    roomId INTEGER,
    PRIMARY KEY (houseId, roomId),
    FOREIGN KEY (houseId) REFERENCES Houses(id) ON DELETE CASCADE,
    FOREIGN KEY (roomId) REFERENCES Rooms(id) ON DELETE CASCADE
);
INSERT INTO "Houses_Rooms" VALUES(1,4);
INSERT INTO "Houses_Rooms" VALUES(2,4);
INSERT INTO "Houses_Rooms" VALUES(3,4);
INSERT INTO "Houses_Rooms" VALUES(4,4);
INSERT INTO "Houses_Rooms" VALUES(5,4);
INSERT INTO "Houses_Rooms" VALUES(5,1);
CREATE TABLE Rooms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roomType TEXT
);
INSERT INTO "Rooms" VALUES(1,'Living Room');
INSERT INTO "Rooms" VALUES(2,'Bedroom');
INSERT INTO "Rooms" VALUES(3,'Dining Room');
INSERT INTO "Rooms" VALUES(4,'Kitchen');
INSERT INTO "Rooms" VALUES(5,'Bathroom');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Rooms',5);
INSERT INTO "sqlite_sequence" VALUES('HouseFeatures',5);
INSERT INTO "sqlite_sequence" VALUES('Houses',5);
COMMIT;
