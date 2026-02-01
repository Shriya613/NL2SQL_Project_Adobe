BEGIN TRANSACTION;
CREATE TABLE Action_Movie(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movieId INTEGER,
    actionDetails TEXT,
    FOREIGN KEY(movieId) REFERENCES Movie(id)
);
INSERT INTO "Action_Movie" VALUES(1,2,'High-octane car chases and explosive stunts.');
INSERT INTO "Action_Movie" VALUES(2,3,'A team of heroes band together to save the world.');
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actorName TEXT UNIQUE
);
INSERT INTO "Actor" VALUES(1,'Alice Johnson');
INSERT INTO "Actor" VALUES(2,'Bob Brown');
INSERT INTO "Actor" VALUES(3,'Charlie Green');
CREATE TABLE Adventure_Movie(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movieId INTEGER,
    adventureDetails TEXT,
    FOREIGN KEY(movieId) REFERENCES Movie(id)
);
INSERT INTO "Adventure_Movie" VALUES(1,1,'A thrilling journey through uncharted lands.');
CREATE TABLE Cast(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movieId INTEGER,
    actorId INTEGER,
    role TEXT,
    FOREIGN KEY(movieId) REFERENCES Movie(id),
    FOREIGN KEY(actorId) REFERENCES Actor(id)
);
INSERT INTO "Cast" VALUES(1,1,1,'Protagonist');
INSERT INTO "Cast" VALUES(2,2,2,'Villain');
INSERT INTO "Cast" VALUES(3,3,3,'Supporting Role');
INSERT INTO "Cast" VALUES(4,1,4,'Sidekick');
CREATE TABLE Director(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Director" VALUES(1,'Sofia Martinez');
INSERT INTO "Director" VALUES(2,'John Doe');
INSERT INTO "Director" VALUES(3,'Jane Smith');
CREATE TABLE Movie(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    directorId INTEGER,
    releaseYear INTEGER,
    FOREIGN KEY(directorId) REFERENCES Director(id)
);
INSERT INTO "Movie" VALUES(1,'The Secret Adventure',1,2023);
INSERT INTO "Movie" VALUES(2,'Racing Hearts',1,2025);
INSERT INTO "Movie" VALUES(3,'Superhero Squad',2,2024);
INSERT INTO "Movie" VALUES(4,'Epic Quest',3,2026);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Director',3);
INSERT INTO "sqlite_sequence" VALUES('Actor',3);
INSERT INTO "sqlite_sequence" VALUES('Movie',4);
INSERT INTO "sqlite_sequence" VALUES('Adventure_Movie',1);
INSERT INTO "sqlite_sequence" VALUES('Action_Movie',2);
INSERT INTO "sqlite_sequence" VALUES('Cast',4);
COMMIT;
