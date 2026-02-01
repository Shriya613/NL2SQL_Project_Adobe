BEGIN TRANSACTION;
CREATE TABLE ActorsInFilms(
    ActorID INTEGER,
    FilmID INTEGER,
    Role TEXT,
    FOREIGN KEY (ActorID) REFERENCES Directors(id),
    FOREIGN KEY (FilmID) REFERENCES FilmsDirectedBy(FilmID));
INSERT INTO "ActorsInFilms" VALUES(1,1,'Himself');
INSERT INTO "ActorsInFilms" VALUES(2,2,'Her');
INSERT INTO "ActorsInFilms" VALUES(3,3,'Leader of the Wolf Pack');
INSERT INTO "ActorsInFilms" VALUES(4,4,'Daughter');
INSERT INTO "ActorsInFilms" VALUES(5,5,'Scientist');
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate TEXT,
    Nationality TEXT);
INSERT INTO "Directors" VALUES(1,'John Smith','1970-01-01','USA');
INSERT INTO "Directors" VALUES(2,'Jane Doe','1980-01-01','UK');
INSERT INTO "Directors" VALUES(3,'Michael Johnson','1965-02-15','Australia');
INSERT INTO "Directors" VALUES(4,'Emily Brown','1985-03-20','Canada');
INSERT INTO "Directors" VALUES(5,'David Williams','1972-04-25','New Zealand');
CREATE TABLE Documentaries(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Genre TEXT,
    Duration REAL,
    Budget TEXT);
INSERT INTO "Documentaries" VALUES(1,'The Silent World','Adventure/Documentary',21600.0,'300000');
INSERT INTO "Documentaries" VALUES(2,'Grizzly Man','Wildlife/Documentary',8975.0,'500000');
INSERT INTO "Documentaries" VALUES(3,'Into The Inferno','Nature/Documentary',15500.0,'750000');
INSERT INTO "Documentaries" VALUES(4,'Blackfish','Animal Rights/Documentary',12800.0,'200000');
INSERT INTO "Documentaries" VALUES(5,'Chasing Ice','Environmental/Documentary',17500.0,'400000');
CREATE TABLE FilmsDirectedBy(
    FilmID INTEGER,
    DirectorID INTEGER,
    FOREIGN KEY (FilmID) REFERENCES Documentaries(id),
    FOREIGN KEY (DirectorID) REFERENCES Directors(id));
INSERT INTO "FilmsDirectedBy" VALUES(1,1);
INSERT INTO "FilmsDirectedBy" VALUES(2,2);
INSERT INTO "FilmsDirectedBy" VALUES(3,3);
INSERT INTO "FilmsDirectedBy" VALUES(4,4);
INSERT INTO "FilmsDirectedBy" VALUES(5,5);
CREATE TABLE ShortFilms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Genre TEXT,
    Duration REAL,
    Budget TEXT);
INSERT INTO "ShortFilms" VALUES(1,'A Day at the Park','Comedy/Short Film',3600.0,'500000');
INSERT INTO "ShortFilms" VALUES(2,'My Best Friend','Drama/Short Film',4200.0,'500000');
INSERT INTO "ShortFilms" VALUES(3,'The Great Escape','Action/Short Film',4800.0,'400000');
INSERT INTO "ShortFilms" VALUES(4,'Lost and Found','Family/Short Film',3300.0,'200000');
INSERT INTO "ShortFilms" VALUES(5,'One Small Step','Science Fiction/Short Film',4500.0,'600000');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Documentaries',5);
INSERT INTO "sqlite_sequence" VALUES('ShortFilms',5);
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
COMMIT;
