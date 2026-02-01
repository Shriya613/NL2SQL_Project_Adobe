BEGIN TRANSACTION;
CREATE TABLE CastMembers (
    castMemberId INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    roleInPerformance TEXT,
    performanceId INTEGER,
    FOREIGN KEY(performanceId) REFERENCES Performances(performanceId)
);
INSERT INTO "CastMembers" VALUES(1,'Jane Doe','Figaro',1);
INSERT INTO "CastMembers" VALUES(2,'John Smith','Count Almaviva',1);
CREATE TABLE Composers (
    composerId INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthYear INTEGER
);
INSERT INTO "Composers" VALUES(1,'Wolfgang Amadeus Mozart',1756);
INSERT INTO "Composers" VALUES(2,'George Frideric Handel',1685);
CREATE TABLE Operas (
    operaID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    Historical_Era TEXT,
    Location TEXT,
    Duration INTEGER,
    composerId INTEGER,
    FOREIGN KEY(composerId) REFERENCES Composers(composerId)
);
INSERT INTO "Operas" VALUES(1,'The Marriage of Figaro','Classical','Seville',300,1);
INSERT INTO "Operas" VALUES(2,'Messiah','Baroque','Jerusalem',220,2);
CREATE TABLE Performances (
    performanceId INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    startTime TEXT,
    venueId INTEGER,
    operaId INTEGER,
    FOREIGN KEY(venueId) REFERENCES Venues(venueId),
    FOREIGN KEY(operaId) REFERENCES Operas(operaId)
);
INSERT INTO "Performances" VALUES(1,'2022-10-15','19:30',1,1);
INSERT INTO "Performances" VALUES(2,'2022-11-20','18:00',2,2);
CREATE TABLE Venues (
    venueID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    capacity INTEGER
);
INSERT INTO "Venues" VALUES(1,'Metropolitan Opera House',3938);
INSERT INTO "Venues" VALUES(2,'Royal Opera House',2100);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Composers',2);
INSERT INTO "sqlite_sequence" VALUES('Operas',2);
INSERT INTO "sqlite_sequence" VALUES('Venues',2);
INSERT INTO "sqlite_sequence" VALUES('Performances',2);
INSERT INTO "sqlite_sequence" VALUES('CastMembers',2);
COMMIT;
