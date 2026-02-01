BEGIN TRANSACTION;
CREATE TABLE Actors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ActorName TEXT,
    Age INT,
    FeePerMovie DECIMAL(10,2),
    MovieId INTEGER,
    FOREIGN KEY(MovieId) REFERENCES Movies(id)
);
INSERT INTO "Actors" VALUES(1,'Tom Hanks',64,20000000,1);
INSERT INTO "Actors" VALUES(2,'Samuel L Jackson',72,1500000,1);
INSERT INTO "Actors" VALUES(3,'Leonardo DiCaprio',47,2500000,2);
INSERT INTO "Actors" VALUES(4,'Kate Winslet',45,1800000,2);
INSERT INTO "Actors" VALUES(5,'Johnny Depp',58,2000000,3);
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    DirectorName TEXT,
    YearsExperience INT
);
INSERT INTO "Directors" VALUES(1,'Steven Spielberg',49);
INSERT INTO "Directors" VALUES(2,'James Cameron',38);
INSERT INTO "Directors" VALUES(3,'Christopher Nolan',26);
INSERT INTO "Directors" VALUES(4,'Quentin Tarantino',32);
INSERT INTO "Directors" VALUES(5,'Martin Scorsese',57);
CREATE TABLE FilmRatings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Rating TEXT,
    Description TEXT
);
INSERT INTO "FilmRatings" VALUES(1,'G','General Audience');
INSERT INTO "FilmRatings" VALUES(2,'PG','Parental Guidance Suggested');
INSERT INTO "FilmRatings" VALUES(3,'PG-13','Special parental guidance suggested for children under 13');
INSERT INTO "FilmRatings" VALUES(4,'R','Restricted - Under 17 requires accompanying parent or adult guardian.');
INSERT INTO "FilmRatings" VALUES(5,'NC-17','No one under 17 admitted');
CREATE TABLE MovieRatings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieId INTEGER,
    RatingId INTEGER,
    FOREIGN KEY(MovieId) REFERENCES Movies(id),
    FOREIGN KEY(RatingId) REFERENCES FilmRatings(id)
);
INSERT INTO "MovieRatings" VALUES(1,1,1);
INSERT INTO "MovieRatings" VALUES(2,2,2);
INSERT INTO "MovieRatings" VALUES(3,3,1);
INSERT INTO "MovieRatings" VALUES(4,4,3);
INSERT INTO "MovieRatings" VALUES(5,5,1);
CREATE TABLE Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieTitle TEXT,
    ReleaseYear INT,
    Genre TEXT,
    ProducerId INTEGER,
    FOREIGN KEY(ProducerId) REFERENCES Producers(id)
);
INSERT INTO "Movies" VALUES(1,'E.T.',1982,'Science-Fiction',1);
INSERT INTO "Movies" VALUES(2,'Avatar',2009,'Adventure/Fantasy',2);
INSERT INTO "Movies" VALUES(3,'Inception',2010,'Thriller',3);
INSERT INTO "Movies" VALUES(4,'Pulp Fiction',1994,'Crime/Drama',4);
INSERT INTO "Movies" VALUES(5,'The Wolf of Wall Street',2013,'Biographical Drama',5);
CREATE TABLE "Producers" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Production TEXT);
INSERT INTO "Producers" VALUES(1,'Steven Spielberg');
INSERT INTO "Producers" VALUES(2,'James Cameron');
INSERT INTO "Producers" VALUES(3,'Christopher Nolan');
INSERT INTO "Producers" VALUES(4,'Quentin Tarantino');
INSERT INTO "Producers" VALUES(5,'Martin Scorsese');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('FilmRatings',5);
INSERT INTO "sqlite_sequence" VALUES('MovieRatings',5);
INSERT INTO "sqlite_sequence" VALUES('Producers',5);
COMMIT;
