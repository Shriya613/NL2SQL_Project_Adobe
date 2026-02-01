BEGIN TRANSACTION;
CREATE TABLE Actors (
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE 
);
INSERT INTO "Actors" VALUES(1,'Christopher','Nolan','1970-07-20');
INSERT INTO "Actors" VALUES(2,'Tom','Hardy','1977-09-15');
INSERT INTO "Actors" VALUES(3,'Matthew','McConaughey','1969-11-04');
INSERT INTO "Actors" VALUES(4,'Anne','Hathaway','1982-11-12');
INSERT INTO "Actors" VALUES(5,'Joseph','Gordon-Levitt','1981-12-17');
CREATE TABLE "Costumes" (
    CostumeID INTEGER PRIMARY KEY AUTOINCREMENT,
    "Costumier" TEXT,
    Costumes TEXT,
    Budget REAL 
);
INSERT INTO "Costumes" VALUES(1,'Eduard Paolantoni','Period costumes and futuristic outfits',800000.0);
INSERT INTO "Costumes" VALUES(2,'Kate Barker','Space suits and alien attire',750000.0);
INSERT INTO "Costumes" VALUES(3,'Alexander Byrne','Modern clothing and uniforms',600000.0);
INSERT INTO "Costumes" VALUES(4,'Colleen Atwood','Victorian dresses and armor',900000.0);
INSERT INTO "Costumes" VALUES(5,'Louise Frogley','Spacesuits and survival gear',850000.0);
CREATE TABLE Directors (
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE 
);
INSERT INTO "Directors" VALUES(1,'Christopher','Nolan','1970-07-20');
CREATE TABLE FilmActors (
    FilmID INTEGER,
    ActorID INTEGER,
    Role TEXT,
    FOREIGN KEY (FilmID) REFERENCES Films(FilmID),
    FOREIGN KEY (ActorID) REFERENCES Actors(ActorID),
    PRIMARY KEY (FilmID, ActorID)
);
CREATE TABLE Films (
    FilmID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INTEGER ,
    Runtime INTEGER 
);
INSERT INTO "Films" VALUES(1,'Inception',2010,160);
INSERT INTO "Films" VALUES(2,'The Dark Knight Rises',2012,180);
INSERT INTO "Films" VALUES(3,'Gravity',2013,90);
INSERT INTO "Films" VALUES(4,'Interstellar',2014,170);
INSERT INTO "Films" VALUES(5,'Avatar',2009,170);
CREATE TABLE Producers (
    ProducerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    ProductionCompany TEXT
);
INSERT INTO "Producers" VALUES(1,'Emma','Thomas','Warner Bros.');
INSERT INTO "Producers" VALUES(2,'Charles','Roven','Legendary Pictures');
INSERT INTO "Producers" VALUES(3,'Lisa','Joy','Blinding Edge Pictures');
INSERT INTO "Producers" VALUES(4,'Deborah','Snyder','Syncopy North');
INSERT INTO "Producers" VALUES(5,'Jonathan','Nolan','Syncopy North');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Films',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Directors',1);
INSERT INTO "sqlite_sequence" VALUES('Producers',5);
INSERT INTO "sqlite_sequence" VALUES('Costumes',5);
COMMIT;
