BEGIN TRANSACTION;
CREATE TABLE Actors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthDate DATE,
    nationality TEXT
);
INSERT INTO "Actors" VALUES(1,'Keanu Reeves','September 2, 1964','Canada');
INSERT INTO "Actors" VALUES(2,'Natalie Portman','June 9, 1981','Israel');
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthDate DATE,
    nationality TEXT
);
INSERT INTO "Directors" VALUES(1,'Christopher Nolan','July 30, 1970','England');
INSERT INTO "Directors" VALUES(2,'Guillermo del Toro','October 9, 1964','Spain');
CREATE TABLE FantasyMovies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    director TEXT,
    releaseYear INTEGER,
    fantasy GENERIC  DEFAULT 'False',
    filmingLocation TEXT
);
INSERT INTO "FantasyMovies" VALUES(1,'Lord of The Rings','Peter Jackson',2001,1,'New Zealand');
INSERT INTO "FantasyMovies" VALUES(2,'Harry Potter and The Philosopher''s Stone','Chris Columbus',2001,1,'United Kingdom');
CREATE TABLE Productions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movieTitle TEXT,
    productionCompany TEXT,
    budget REAL,
    startDate DATE,
    endDate DATE
);
INSERT INTO "Productions" VALUES(1,'Inception','Warner Bros.',160.0,'June 16, 2009','August 16, 2010');
INSERT INTO "Productions" VALUES(2,'Pan''s Labyrinth','Pictaza',30.0,'April 17, 2005','December 17, 2005');
CREATE TABLE Roles(
    movieId INTEGER,
    actorId INTEGER,
    roleName TEXT,
    FOREIGN KEY(movieId) REFERENCES Movies(id),
    FOREIGN KEY(actorId) REFERENCES Actors(id),
    PRIMARY KEY(movieId, actorId)
);
INSERT INTO "Roles" VALUES(1,1,'Neo');
INSERT INTO "Roles" VALUES(3,2,'Padmé Amidala');
CREATE TABLE SciFiMovies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    director TEXT,
    releaseYear INTEGER,
    sciFi GENERIC  DEFAULT 'False',
    filmingLocation TEXT
);
INSERT INTO "SciFiMovies" VALUES(1,'Star Wars','George Lucas',1977,1,'New Zealand');
INSERT INTO "SciFiMovies" VALUES(2,'Blade Runner','Ridley Scott',1982,1,'United States');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('SciFiMovies',2);
INSERT INTO "sqlite_sequence" VALUES('FantasyMovies',2);
INSERT INTO "sqlite_sequence" VALUES('Actors',2);
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
INSERT INTO "sqlite_sequence" VALUES('Productions',2);
COMMIT;
