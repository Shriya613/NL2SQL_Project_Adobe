BEGIN TRANSACTION;
CREATE TABLE Broadsheet (
    NewspaperID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    Price REAL,
    DistributorID INTEGER,
    FOREIGN KEY(DistributorID) REFERENCES Distributor(ID)
);
INSERT INTO "Broadsheet" VALUES(1,'The Guardian','Latest news and analysis...',1.2,1);
INSERT INTO "Broadsheet" VALUES(2,'Financial Times','Business and financial news...',3.0,1);
INSERT INTO "Broadsheet" VALUES(3,'Independent','News and opinion articles...',1.5,2);
INSERT INTO "Broadsheet" VALUES(4,'Telegraph','Current affairs and commentary...',2.0,2);
INSERT INTO "Broadsheet" VALUES(5,'Times','UK and world news updates...',3.5,1);
CREATE TABLE Distributor (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Distributor" VALUES(1,'Sainsbury''s');
INSERT INTO "Distributor" VALUES(2,'Tesco');
CREATE TABLE Tabloid (
    NewspaperID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    Price REAL,
    DistributorID INTEGER,
    FOREIGN KEY(DistributorID) REFERENCES Distributor(ID)
);
INSERT INTO "Tabloid" VALUES(1,'Sun','Celebrity gossip and sports...',0.8,1);
INSERT INTO "Tabloid" VALUES(2,'Mirror','General interest stories and showbiz news...',0.6,1);
INSERT INTO "Tabloid" VALUES(3,'Express','Conservative politics and news...',0.9,2);
INSERT INTO "Tabloid" VALUES(4,'Star','Showbiz, entertainment, and celebrity news...',0.7,2);
INSERT INTO "Tabloid" VALUES(5,'Mail','General news with conservative bias...',0.95,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Distributor',2);
INSERT INTO "sqlite_sequence" VALUES('Broadsheet',5);
INSERT INTO "sqlite_sequence" VALUES('Tabloid',5);
COMMIT;
