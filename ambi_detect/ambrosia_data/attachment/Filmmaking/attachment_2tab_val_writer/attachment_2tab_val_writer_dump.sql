BEGIN TRANSACTION;
CREATE TABLE Actors(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    BirthDate DATE,
    Nationality TEXT,
    FamousFor TEXT
);
INSERT INTO "Actors" VALUES(1,'Tom Hanks','1956-07-09','American','Forest Gump');
INSERT INTO "Actors" VALUES(2,'Meryl Streep','1949-06-22','American','The Iron Lady');
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    BirthDate DATE,
    Nationality TEXT,
    FavoriteGenre TEXT
);
INSERT INTO "Directors" VALUES(1,'David Miller','1978-11-12','French','Drama');
INSERT INTO "Directors" VALUES(2,'Emily Green','1982-02-25','Irish','Comedy');
CREATE TABLE FilmProductions(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    ReleaseYear INTEGER,
    Budget REAL,
    BoxOffice REAL
);
INSERT INTO "FilmProductions" VALUES(1,'Inception',2010,160000000.0,8283700000.0);
INSERT INTO "FilmProductions" VALUES(2,'Avatar',2009,237000000.0,28472300000.0);
CREATE TABLE Playwrights(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    BirthDate DATE,
    Nationality TEXT,
    Education TEXT
);
INSERT INTO "Playwrights" VALUES(1,'Mary Johnson','1990-03-07','Canadian','New York University');
INSERT INTO "Playwrights" VALUES(2,'William Brown','1985-08-21','Australian','University of Sydney');
CREATE TABLE Screenwriters(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    BirthDate DATE,
    Nationality TEXT,
    Education TEXT
);
INSERT INTO "Screenwriters" VALUES(1,'John Doe','1980-06-23','American','New York University');
INSERT INTO "Screenwriters" VALUES(2,'Jane Smith','1975-10-15','British','University of Cambridge');
COMMIT;
