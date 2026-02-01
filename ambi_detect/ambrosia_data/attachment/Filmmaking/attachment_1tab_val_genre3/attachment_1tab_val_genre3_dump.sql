BEGIN TRANSACTION;
CREATE TABLE Actors(
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE
);
INSERT INTO "Actors" VALUES(1,'Tom','Hanks','1956-07-09');
INSERT INTO "Actors" VALUES(2,'Emma','Stone','1988-11-06');
INSERT INTO "Actors" VALUES(3,'Leonardo','DiCaprio','1974-11-11');
CREATE TABLE Directors(
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE
);
INSERT INTO "Directors" VALUES(1,'John Doe','1985-07-06');
INSERT INTO "Directors" VALUES(2,'Jane Smith','1990-03-12');
INSERT INTO "Directors" VALUES(3,'Alice Brown','1975-10-24');
CREATE TABLE Movies(
    MovieID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Genre TEXT,
    Budget TEXT,
    ReleaseYear INT,
    DirectorID INTEGER,
    FOREIGN KEY(DirectorID) REFERENCES Directors(DirectorID)
);
INSERT INTO "Movies" VALUES(1,'Scary Night','Horror','20 million USD',2010,1);
INSERT INTO "Movies" VALUES(2,'Silent Shadows','Horror','15 million USD',2005,2);
INSERT INTO "Movies" VALUES(3,'Chilling Escape','Thriller','20 million USD',2009,3);
INSERT INTO "Movies" VALUES(4,'Twisted Minds','Thriller','10 million USD',2015,3);
INSERT INTO "Movies" VALUES(5,'Action Adventure','Adventure','12 million USD',2012,1);
INSERT INTO "Movies" VALUES(6,'Fantasy Quest','Fantasy','15 million USD',2011,2);
CREATE TABLE Roles(
    RoleID INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieID INTEGER,
    ActorID INTEGER,
    CharacterName TEXT,
    FOREIGN KEY(MovieID) REFERENCES Movies(MovieID),
    FOREIGN KEY(ActorID) REFERENCES Actors(ActorID)
);
INSERT INTO "Roles" VALUES(1,1,1,'Jack');
INSERT INTO "Roles" VALUES(2,2,1,'Samantha');
INSERT INTO "Roles" VALUES(3,3,2,'David');
INSERT INTO "Roles" VALUES(4,4,3,'Lucy');
INSERT INTO "Roles" VALUES(5,5,1,'Kevin');
INSERT INTO "Roles" VALUES(6,6,3,'Rachel');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',3);
INSERT INTO "sqlite_sequence" VALUES('Movies',6);
INSERT INTO "sqlite_sequence" VALUES('Actors',3);
INSERT INTO "sqlite_sequence" VALUES('Roles',6);
COMMIT;
