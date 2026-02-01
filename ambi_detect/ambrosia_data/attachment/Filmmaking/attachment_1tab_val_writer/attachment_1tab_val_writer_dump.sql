BEGIN TRANSACTION;
CREATE TABLE Director (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE
);
INSERT INTO "Director" VALUES(1,'Christopher Nolan','30 July 1970');
INSERT INTO "Director" VALUES(2,'Lana Wachowski','21 June 1968');
INSERT INTO "Director" VALUES(3,'David Fincher','August 22, 1962');
INSERT INTO "Director" VALUES(4,'Quentin Tarantino','March 27, 1963');
INSERT INTO "Director" VALUES(5,'Martin Scorsese','November 17, 1942');
CREATE TABLE Director_Film (
    directorId INTEGER,
    filmId INTEGER,
    FOREIGN KEY(directorId) REFERENCES Director(id),
    FOREIGN KEY(filmId) REFERENCES Film(id),
    PRIMARY KEY(directorId, filmId)
);
INSERT INTO "Director_Film" VALUES(1,1);
INSERT INTO "Director_Film" VALUES(2,2);
INSERT INTO "Director_Film" VALUES(3,3);
INSERT INTO "Director_Film" VALUES(4,4);
INSERT INTO "Director_Film" VALUES(5,5);
CREATE TABLE Film (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INTEGER,
    Genre TEXT
);
INSERT INTO "Film" VALUES(1,'Inception',2010,'Science Fiction');
INSERT INTO "Film" VALUES(2,'The Matrix',1999,'Science Fiction');
INSERT INTO "Film" VALUES(3,'Fight Club',1999,'Drama');
INSERT INTO "Film" VALUES(4,'Pulp Fiction',1994,'Crime');
INSERT INTO "Film" VALUES(5,'Goodfellas',1990,'Crime');
CREATE TABLE Film_Screenplay (
    filmId INTEGER,
    scriptId INTEGER,
    FOREIGN KEY(scriptId) REFERENCES Script(id),
    FOREIGN KEY(filmId) REFERENCES Film(id),
    PRIMARY KEY(filmId, scriptId)
);
INSERT INTO "Film_Screenplay" VALUES(1,1);
INSERT INTO "Film_Screenplay" VALUES(2,2);
INSERT INTO "Film_Screenplay" VALUES(3,3);
INSERT INTO "Film_Screenplay" VALUES(4,4);
INSERT INTO "Film_Screenplay" VALUES(5,5);
CREATE TABLE Screenplay (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Year WRITER INT
);
INSERT INTO "Screenplay" VALUES(1,'Inception',1);
INSERT INTO "Screenplay" VALUES(2,'The Matrix',2);
INSERT INTO "Screenplay" VALUES(3,'Fight Club',3);
INSERT INTO "Screenplay" VALUES(4,'Pulp Fiction',4);
INSERT INTO "Screenplay" VALUES(5,'Goodfellas',5);
CREATE TABLE Writer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Type TEXT, -- Could be either 'screenwriter', 'playwright', or others
    Education TEXT
);
INSERT INTO "Writer" VALUES(1,'John Doe','Screenwriter','New York University');
INSERT INTO "Writer" VALUES(2,'Jane Smith','Screenwriter','University of California Los Angeles');
INSERT INTO "Writer" VALUES(3,'Bob Johnson','Playwright','New York University');
INSERT INTO "Writer" VALUES(4,'Alice Brown','Playwright','Columbia University');
INSERT INTO "Writer" VALUES(5,'Tom Green','Screenwriter','American Film Institute');
INSERT INTO "Writer" VALUES(6,'Mary White','Director','UCLA School of Theater, Film and Television');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Writer',6);
INSERT INTO "sqlite_sequence" VALUES('Film',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Screenplay',5);
COMMIT;
