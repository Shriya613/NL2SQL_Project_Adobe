BEGIN TRANSACTION;
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Birthdate DATE,
    Nationality TEXT
);
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Birthdate DATE,
    Nationality TEXT
);
CREATE TABLE Genres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT
);
INSERT INTO "Genres" VALUES(1,'Action');
INSERT INTO "Genres" VALUES(2,'Drama');
CREATE TABLE MovieGenre (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieId INTEGER,
    GenreId INTEGER,
    FOREIGN KEY(MovieId) REFERENCES Movies(id),
    FOREIGN KEY(GenreId) REFERENCES Genres(id)
);
INSERT INTO "MovieGenre" VALUES(1,1,2);
INSERT INTO "MovieGenre" VALUES(2,2,NULL);
CREATE TABLE Movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseDate DATE,
    Runtime INT,
    Filmmaker TEXT,
    Screenwriter TEXT
);
INSERT INTO "Movies" VALUES(1,'The Shawshank Redemption','1994-09-23',142,'Frank Darabont','Stephen King');
INSERT INTO "Movies" VALUES(2,'Pulp Fiction','1994-10-14',154,'Quentin Tarantino','Quentin Tarantino');
CREATE TABLE Productions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieId INTEGER,
    DirectorId INTEGER,
    Budget INT,
    ProductionCompany TEXT,
    FOREIGN KEY(MovieId) REFERENCES Movies(id),
    FOREIGN KEY(DirectorId) REFERENCES Directors(id)
);
CREATE TABLE Roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieId INTEGER,
    ActorId INTEGER,
    CharacterName TEXT,
    FOREIGN KEY(MovieId) REFERENCES Movies(id),
    FOREIGN KEY(ActorId) REFERENCES Actors(id)
);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Movies',2);
INSERT INTO "sqlite_sequence" VALUES('Genres',2);
INSERT INTO "sqlite_sequence" VALUES('MovieGenre',2);
COMMIT;
