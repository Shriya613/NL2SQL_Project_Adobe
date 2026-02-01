BEGIN TRANSACTION;
CREATE TABLE Actors (
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Birthdate DATE
);
INSERT INTO "Actors" VALUES(1,'John','Doe','1980-06-23');
INSERT INTO "Actors" VALUES(2,'Jane','Smith','1978-05-12');
INSERT INTO "Actors" VALUES(3,'Emma','Wilson','1992-04-01');
INSERT INTO "Actors" VALUES(4,'Tom','Brown','1995-03-16');
INSERT INTO "Actors" VALUES(5,'Alice','Green','1990-07-29');
CREATE TABLE Cast (
    ProductionID INTEGER,
    ActorID INTEGER,
    Role TEXT,
    FOREIGN KEY(ProductionID) REFERENCES Productions(ProductionID),
    FOREIGN KEY(ActorID) REFERENCES Actors(ActorID),
    PRIMARY KEY(ProductionID, ActorID)
);
INSERT INTO "Cast" VALUES(1,1,'Lead Actor');
INSERT INTO "Cast" VALUES(1,2,'Supporting Actor');
INSERT INTO "Cast" VALUES(2,3,'Lead Actor');
INSERT INTO "Cast" VALUES(2,4,'Supporting Actor');
INSERT INTO "Cast" VALUES(3,5,'Narrator');
CREATE TABLE Locations (
    LocationID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Type TEXT
);
INSERT INTO "Locations" VALUES(1,'Studio Sets 1','Studio_Set');
INSERT INTO "Locations" VALUES(2,'Exterior Shoots 1','Exterior');
INSERT INTO "Locations" VALUES(3,'Interior Scenes 1','Interior');
INSERT INTO "Locations" VALUES(4,'Sound Stage 1','Stage');
INSERT INTO "Locations" VALUES(5,'Backlot 1','Backlot');
CREATE TABLE Productions (
    ProductionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseDate DATE
);
INSERT INTO "Productions" VALUES(1,'Movie One','2021-01-01');
INSERT INTO "Productions" VALUES(2,'Film Two','2021-05-01');
INSERT INTO "Productions" VALUES(3,'Series Three','2021-07-01');
INSERT INTO "Productions" VALUES(4,'Documentary Four','2021-09-01');
INSERT INTO "Productions" VALUES(5,'Show Five','2021-11-01');
CREATE TABLE Productions_Locations (
    ProductionID INTEGER,
    LocationID INTEGER,
    FOREIGN KEY(ProductionID) REFERENCES Productions(ProductionID),
    FOREIGN KEY(LocationID) REFERENCES Locations(LocationID),
    PRIMARY KEY(ProductionID, LocationID)
);
INSERT INTO "Productions_Locations" VALUES(1,1);
INSERT INTO "Productions_Locations" VALUES(2,1);
INSERT INTO "Productions_Locations" VALUES(3,1);
INSERT INTO "Productions_Locations" VALUES(4,1);
INSERT INTO "Productions_Locations" VALUES(5,1);
INSERT INTO "Productions_Locations" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Productions',5);
INSERT INTO "sqlite_sequence" VALUES('Locations',5);
COMMIT;
