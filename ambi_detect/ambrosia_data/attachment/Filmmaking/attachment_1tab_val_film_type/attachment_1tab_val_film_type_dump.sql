BEGIN TRANSACTION;
CREATE TABLE Actors(
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Actors" VALUES(1,'Tom','Hanks','1956-07-09');
INSERT INTO "Actors" VALUES(2,'Meryl','Streep','1949-06-22');
CREATE TABLE Casts(
    RoleID INTEGER PRIMARY KEY AUTOINCREMENT,
    FilmID INTEGER,
    ActorID INTEGER,
    CharacterName TEXT,
    AppearanceFee DECIMAL(10,2),
    FOREIGN KEY (ActorID) REFERENCES Actors(ActorID),
    FOREIGN KEY (FilmID) REFERENCES Films(FilmID)
);
INSERT INTO "Casts" VALUES(1,1,1,'Main Lead',100000);
INSERT INTO "Casts" VALUES(2,2,2,'Supporting Role',50000);
CREATE TABLE Companies(
    CompanyID INTEGER PRIMARY KEY AUTOINCREMENT,
    CompanyName TEXT,
    AddressLine1 TEXT,
    AddressLine2 TEXT,
    City TEXT,
    StateOrRegion TEXT,
    PostalCode TEXT,
    Country TEXT
);
INSERT INTO "Companies" VALUES(1,'Big Movie Co.','123 Main St.','','New York','NY','10001','USA');
INSERT INTO "Companies" VALUES(2,'Small Indie Studio','456 Elm Rd.','','Los Angeles','CA','90001','USA');
CREATE TABLE Directors(
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Nationality TEXT
);
INSERT INTO "Directors" VALUES(1,'John Doe','1968-07-23','American');
INSERT INTO "Directors" VALUES(2,'Jane Smith','1970-03-10','Canadian');
CREATE TABLE Films(
    FilmID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INTEGER,
    FilmType TEXT ,
    Budget REAL,
    ProductionCompanyID INTEGER,
    FOREIGN KEY (ProductionCompanyID) REFERENCES Companies(CompanyID)
);
INSERT INTO "Films" VALUES(1,'The Documentary',2000,'Documentary',500000.0,1);
INSERT INTO "Films" VALUES(2,'Another Doc',2001,'Documentary',750000.0,1);
INSERT INTO "Films" VALUES(3,'A Short Film',2002,'Short Film',500000.0,2);
INSERT INTO "Films" VALUES(4,'Yet Another Short',2003,'Short Film',300000.0,2);
CREATE TABLE Producers(
    ProducerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    ContactNumber TEXT,
    Email TEXT,
    AssociatedDirector INTEGER,
    FOREIGN KEY (AssociatedDirector) REFERENCES Directors(DirectorID)
);
INSERT INTO "Producers" VALUES(1,'Alice Brown','+1234567890','ab@email.com',1);
INSERT INTO "Producers" VALUES(2,'Charlie Green','+0987654321','cg@email.com',2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
INSERT INTO "sqlite_sequence" VALUES('Producers',2);
INSERT INTO "sqlite_sequence" VALUES('Films',4);
INSERT INTO "sqlite_sequence" VALUES('Actors',2);
INSERT INTO "sqlite_sequence" VALUES('Casts',2);
INSERT INTO "sqlite_sequence" VALUES('Companies',2);
COMMIT;
