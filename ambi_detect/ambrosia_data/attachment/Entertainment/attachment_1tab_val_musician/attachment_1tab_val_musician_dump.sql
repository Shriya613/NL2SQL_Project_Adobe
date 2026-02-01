BEGIN TRANSACTION;
CREATE TABLE Album(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    ReleaseYear INTEGER,
    ArtistId INTEGER,
    FOREIGN KEY(ArtistId) REFERENCES Musician(id));
INSERT INTO "Album" VALUES(1,'Debut',2010,1);
INSERT INTO "Album" VALUES(2,'Second Act',2015,2);
CREATE TABLE Concert(
    id INTEGER PRIMARY KEY,
    Date DATETIME,
    Venue TEXT,
    HeadlinerId INTEGER,
    FOREIGN KEY(HeadlinerId) REFERENCES Musician(id));
INSERT INTO "Concert" VALUES(1,'2022-06-10','Stadium A',1);
INSERT INTO "Concert" VALUES(2,'2022-07-15','Arena B',3);
CREATE TABLE Fan(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE);
INSERT INTO "Fan" VALUES(1,'Fan1','Lastname1','fan1@email.com');
INSERT INTO "Fan" VALUES(2,'Fan2','Lastname2','fan2@email.com');
CREATE TABLE Musician(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Genre TEXT,
    Instrument TEXT,
    YearOfBirth INTEGER);
INSERT INTO "Musician" VALUES(1,'John Doe','Country Singer','Guitar',1980);
INSERT INTO "Musician" VALUES(2,'Jane Smith','Country Singer','Piano',1975);
INSERT INTO "Musician" VALUES(3,'Tom Brown','Bluegrass Singer','Guitar',1965);
INSERT INTO "Musician" VALUES(4,'Ann White','Bluegrass Singer','Violin',1982);
INSERT INTO "Musician" VALUES(5,'Rock Star','Rocker','Guitar',1990);
INSERT INTO "Musician" VALUES(6,'Pop Queen','Pop Star','Vocals',1993);
CREATE TABLE Purchase(
    id INTEGER PRIMARY KEY,
    AmountMoney REAL,
    ConcertId INTEGER,
    FanId INTEGRER,
    FOREIGN KEY(ConcertId) REFERENCES Concert(id),
    FOREIGN KEY(FanId) REFERENCES Fan(id));
INSERT INTO "Purchase" VALUES(1,100.0,1,1);
INSERT INTO "Purchase" VALUES(2,150.0,2,2);
CREATE TABLE Song(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Duration TIME,
    AlbumId INTEGER,
    FOREIGN KEY(AlbumId) REFERENCES Album(id));
INSERT INTO "Song" VALUES(1,'First Single','03:30',1);
INSERT INTO "Song" VALUES(2,'Second Track','04:00',2);
COMMIT;
