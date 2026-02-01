BEGIN TRANSACTION;
CREATE TABLE Actors(
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Nationality TEXT
);
INSERT INTO "Actors" VALUES(1,'John','Doe','1972-03-15','USA');
INSERT INTO "Actors" VALUES(2,'Jane','Smith','1985-06-23','UK');
CREATE TABLE Directors(
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    Nationality TEXT
);
INSERT INTO "Directors" VALUES(1,'William','Ramirez','1965-08-10','USA');
INSERT INTO "Directors" VALUES(2,'Emma','Thompson','1965-12-15','UK');
CREATE TABLE ShowDirectors(
    ShowID INTEGER,
    DirectorID INTEGER,
    FOREIGN KEY(ShowID) REFERENCES TelevisionShows(ID),
    FOREIGN KEY(DirectorID) REFERENCES Directors(DirectorID)
);
INSERT INTO "ShowDirectors" VALUES(1,1);
INSERT INTO "ShowDirectors" VALUES(2,2);
CREATE TABLE ShowsActors(
    ShowID INTEGER,
    ActorID INTEGER,
    Role TEXT,
    FOREIGN KEY(ShowID) REFERENCES TelevisionShows(ID),
    FOREIGN KEY(ActorID) REFERENCES Actors(ActorID)
);
INSERT INTO "ShowsActors" VALUES(1,1,'Host');
INSERT INTO "ShowsActors" VALUES(1,2,'Co-host');
CREATE TABLE TelevisionShows(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Genre TEXT,
    Country TEXT,
    Television_Format TEXT,
    Audience_Score REAL
);
INSERT INTO "TelevisionShows" VALUES(1,'MasterChef','Cooking Show','USA','TV Series',7.5);
INSERT INTO "TelevisionShows" VALUES(2,'Top Chef','Cooking Show','USA','TV Series',8.0);
INSERT INTO "TelevisionShows" VALUES(3,'Survivor','Reality TV Show','USA','TV Series',7.5);
INSERT INTO "TelevisionShows" VALUES(4,'Big Brother','Reality TV Show','USA','TV Series',6.9);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('TelevisionShows',4);
INSERT INTO "sqlite_sequence" VALUES('Actors',2);
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
COMMIT;
