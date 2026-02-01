BEGIN TRANSACTION;
CREATE TABLE Facilities(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    facilityName TEXT,
    description TEXT
);
INSERT INTO "Facilities" VALUES(1,'Restrooms','Clean restroom facilities available.');
INSERT INTO "Facilities" VALUES(2,'Dining Hall','Variety of food options available.');
INSERT INTO "Facilities" VALUES(3,'Shopping Center','Multiple shopping options available.');
INSERT INTO "Facilities" VALUES(4,'Business Lounge','Relaxing lounge area for business travelers.');
INSERT INTO "Facilities" VALUES(5,'Kids Play Area','Safe play area for children.');
CREATE TABLE Terminals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Terminals" VALUES(1,'Terminal A','Gate 1');
INSERT INTO "Terminals" VALUES(2,'Terminal B','Gate 2');
INSERT INTO "Terminals" VALUES(3,'Terminal C','Gate 3');
INSERT INTO "Terminals" VALUES(4,'Terminal D','Gate 4');
INSERT INTO "Terminals" VALUES(5,'Terminal E','Gate 5');
CREATE TABLE Terminals_Facilities(
    terminalId INTEGER,
    facilityId INTEGER,
    FOREIGN KEY (terminalId) REFERENCES Terminals(id),
    FOREIGN KEY (facilityId) REFERENCES Facilities(id),
    PRIMARY KEY (terminalId, facilityId)
);
INSERT INTO "Terminals_Facilities" VALUES(1,1);
INSERT INTO "Terminals_Facilities" VALUES(1,2);
INSERT INTO "Terminals_Facilities" VALUES(2,1);
INSERT INTO "Terminals_Facilities" VALUES(2,3);
INSERT INTO "Terminals_Facilities" VALUES(3,4);
INSERT INTO "Terminals_Facilities" VALUES(3,5);
INSERT INTO "Terminals_Facilities" VALUES(4,1);
INSERT INTO "Terminals_Facilities" VALUES(4,2);
INSERT INTO "Terminals_Facilities" VALUES(5,3);
INSERT INTO "Terminals_Facilities" VALUES(5,5);
INSERT INTO "Terminals_Facilities" VALUES(3,1);
INSERT INTO "Terminals_Facilities" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Terminals',5);
INSERT INTO "sqlite_sequence" VALUES('Facilities',5);
COMMIT;
