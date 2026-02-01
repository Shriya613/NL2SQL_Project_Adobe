BEGIN TRANSACTION;
CREATE TABLE Actors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Actors" VALUES(1,'Tom','Hanks');
INSERT INTO "Actors" VALUES(2,'Meryl','Streep');
INSERT INTO "Actors" VALUES(3,'Leonardo','DiCaprio');
INSERT INTO "Actors" VALUES(4,'Jennifer','Lawrence');
INSERT INTO "Actors" VALUES(5,'Robert','Downey Jr.');
CREATE TABLE Directors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Directors" VALUES(1,'Martin','Scorsese');
INSERT INTO "Directors" VALUES(2,'Steven','Spielberg');
INSERT INTO "Directors" VALUES(3,'Christopher','Nolan');
INSERT INTO "Directors" VALUES(4,'Quentin','Tarantino');
INSERT INTO "Directors" VALUES(5,'Baz','Luhrmann');
CREATE TABLE Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    directorId INTEGER,
    FOREIGN KEY (directorId) REFERENCES Directors(id)
);
CREATE TABLE Plays(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    genre TEXT
);
INSERT INTO "Plays" VALUES(1,'Hamlet','Tragedy');
INSERT INTO "Plays" VALUES(2,'Macbeth','Tragedy');
INSERT INTO "Plays" VALUES(3,'A Midsummer Night''s Dream','Comedy');
INSERT INTO "Plays" VALUES(4,'Othello','Tragedy');
INSERT INTO "Plays" VALUES(5,'King Lear','Tragedy');
CREATE TABLE TheatrePerformances(
    theatreId INTEGER,
    playId INTEGER,
    performanceDate DATE,
    actorId INTEGER,
    directorId INTEGER,
    FOREIGN KEY (theatreId) REFERENCES Theatres(id),
    FOREIGN KEY (playId) REFERENCES Plays(id),
    FOREIGN KEY (actorId) REFERENCES Actors(id),
    FOREIGN KEY (directorId) REFERENCES Directors(id),
    PRIMARY KEY (theatreId, playId, performanceDate, actorId, directorId)
);
INSERT INTO "TheatrePerformances" VALUES(1,1,'2023-07-01',1,1);
INSERT INTO "TheatrePerformances" VALUES(1,2,'2023-07-15',2,2);
INSERT INTO "TheatrePerformances" VALUES(2,1,'2023-08-01',3,3);
INSERT INTO "TheatrePerformances" VALUES(2,2,'2023-08-15',4,4);
INSERT INTO "TheatrePerformances" VALUES(3,1,'2023-09-01',5,5);
INSERT INTO "TheatrePerformances" VALUES(4,1,'2023-09-01',2,3);
INSERT INTO "TheatrePerformances" VALUES(5,1,'2023-09-01',3,2);
CREATE TABLE Theatres(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Theatres" VALUES(1,'Broadway Theatre','New York City');
INSERT INTO "Theatres" VALUES(2,'Shakespeare Theatre','Washington DC');
INSERT INTO "Theatres" VALUES(3,'Globe Theatre','London');
INSERT INTO "Theatres" VALUES(4,'Imperial Theatre','Toronto');
INSERT INTO "Theatres" VALUES(5,'National Theatre','Paris');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Plays',5);
INSERT INTO "sqlite_sequence" VALUES('Theatres',5);
COMMIT;
