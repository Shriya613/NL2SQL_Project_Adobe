BEGIN TRANSACTION;
CREATE TABLE Airlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    country TEXT
);
INSERT INTO "Airlines" VALUES(1,'Air France','France');
INSERT INTO "Airlines" VALUES(2,'KLM Royal Dutch Airlines','Netherlands');
INSERT INTO "Airlines" VALUES(3,'Delta Air Lines','United States');
INSERT INTO "Airlines" VALUES(4,'Emirates','UAE');
INSERT INTO "Airlines" VALUES(5,'Lufthansa','Germany');
CREATE TABLE Airlines_Fleets (
    airlineId INTEGER REFERENCES Airlines(id),
    fleetId INTEGER REFERENCES Fleets(id),
    FOREIGN KEY (airlineId) REFERENCES Airlines(id),
    FOREIGN KEY (fleetId) REFERENCES Fleets(id),
    PRIMARY KEY (airlineId, fleetId)
);
INSERT INTO "Airlines_Fleets" VALUES(1,1);
INSERT INTO "Airlines_Fleets" VALUES(2,1);
INSERT INTO "Airlines_Fleets" VALUES(3,1);
INSERT INTO "Airlines_Fleets" VALUES(4,1);
INSERT INTO "Airlines_Fleets" VALUES(5,1);
INSERT INTO "Airlines_Fleets" VALUES(5,5);
CREATE TABLE Fleets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fleetName TEXT,
    capacity INTEGER
);
INSERT INTO "Fleets" VALUES(1,'Boeing_737',200);
INSERT INTO "Fleets" VALUES(2,'Airbus A380',517);
INSERT INTO "Fleets" VALUES(3,'Bombardier CRJ-900',76);
INSERT INTO "Fleets" VALUES(4,'Embraer E190',100);
INSERT INTO "Fleets" VALUES(5,'De Havilland Canada Dash 8 Q400',78);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Fleets',5);
COMMIT;
