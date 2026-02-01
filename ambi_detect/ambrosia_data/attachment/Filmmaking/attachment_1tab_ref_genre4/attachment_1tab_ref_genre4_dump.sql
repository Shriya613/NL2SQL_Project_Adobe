BEGIN TRANSACTION;
CREATE TABLE Actors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthdate DATE
);
INSERT INTO "Actors" VALUES(1,'Alice Johnson','1990-03-03');
INSERT INTO "Actors" VALUES(2,'Bob Brown','1985-04-04');
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthdate DATE
);
INSERT INTO "Directors" VALUES(1,'John Doe','1970-01-01');
INSERT INTO "Directors" VALUES(2,'Jane Smith','1980-02-02');
CREATE TABLE Films(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseDate DATE,
    directorId INTEGER,
    genreId INTEGER,
    filmingLocation TEXT,
    FOREIGN KEY(directorId) REFERENCES Directors(id),
    FOREIGN KEY(genreId) REFERENCES Genres(id)
);
INSERT INTO "Films" VALUES(1,'Movie1','2020-01-01',1,1,'New Zealand');
INSERT INTO "Films" VALUES(2,'Movie2','2021-02-02',2,2,'Australia');
INSERT INTO "Films" VALUES(3,'Movie3','2019-03-03',1,2,'New Zealand');
INSERT INTO "Films" VALUES(4,'Movie4','2022-04-04',2,1,'UK');
CREATE TABLE Genres(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    genreName TEXT UNIQUE
);
INSERT INTO "Genres" VALUES(1,'Sci-Fi');
INSERT INTO "Genres" VALUES(2,'Fantasy');
CREATE TABLE Roles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movieId INTEGER,
    actorId INTEGER,
    roleName TEXT,
    FOREIGN KEY(actorId) REFERENCES Actors(id),
    FOREIGN KEY(movieId) REFERENCES Movies(id)
);
INSERT INTO "Roles" VALUES(1,1,1,'Lead Actor');
INSERT INTO "Roles" VALUES(2,2,2,'Supporting Actor');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
INSERT INTO "sqlite_sequence" VALUES('Actors',2);
INSERT INTO "sqlite_sequence" VALUES('Roles',2);
INSERT INTO "sqlite_sequence" VALUES('Genres',2);
INSERT INTO "sqlite_sequence" VALUES('Films',4);
COMMIT;
