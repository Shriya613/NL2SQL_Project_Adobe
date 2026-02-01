BEGIN TRANSACTION;
CREATE TABLE Actor (
    actor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE,
    nationality TEXT
);
INSERT INTO "Actor" VALUES(1,'Tom','Hanks','1956-07-09','American');
INSERT INTO "Actor" VALUES(2,'Meryl','Streep','1949-06-22','American');
CREATE TABLE Director (
    director_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE,
    nationality TEXT
);
INSERT INTO "Director" VALUES(1,'Robert','Zemeckis','1951-05-14','American');
INSERT INTO "Director" VALUES(2,'Francis','Ford Coppola','1939-04-07','American');
CREATE TABLE Movie (
    movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    box_office REAL
);
INSERT INTO "Movie" VALUES(1,'Forrest Gump',1994,'Drama',6773847284.0);
INSERT INTO "Movie" VALUES(2,'The Godfather',1972,'Crime',245124240.0);
CREATE TABLE Performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    play_id INTEGER,
    theatre_id INTEGER,
    date DATE,
    time TIME,
    ticket_price REAL,
    FOREIGN KEY(play_id) REFERENCES Play(play_id),
    FOREIGN KEY(theatre_id) REFERENCES Theatre(theatre_id)
);
INSERT INTO "Performance" VALUES(1,1,1,'2023-01-01','19:00:00',100.0);
INSERT INTO "Performance" VALUES(2,1,2,'2023-01-01','19:00:00',100.0);
CREATE TABLE "Play" (
    play_id INTEGER PRIMARY KEY AUTOINCREMENT,
    Context INTEGER,
    title TEXT,
    author TEXT,
    genre TEXT,
    length INTEGER
);
INSERT INTO "Play" VALUES(1,1606,'Macbeth','William Shakespeare','Tragedy',200);
INSERT INTO "Play" VALUES(2,1897,'Uncle Vanya','Anton Chekhov','Tragicomedy',150);
CREATE TABLE Role (
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id INTEGER,
    movie_id INTEGER,
    character_name TEXT,
    FOREIGN KEY(actor_id) REFERENCES Actor(actor_id),
    FOREIGN KEY(movie_id) REFERENCES Movie(movie_id)
);
INSERT INTO "Role" VALUES(1,1,1,'Forest Gump');
INSERT INTO "Role" VALUES(2,2,1,'Jenny Curran');
CREATE TABLE Theatre (
    theatre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    capacity INTEGER
);
INSERT INTO "Theatre" VALUES(1,'Broadway Theatre','New York City',2000);
INSERT INTO "Theatre" VALUES(2,'Apollo Theatre','London',1000);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Actor',2);
INSERT INTO "sqlite_sequence" VALUES('Movie',2);
INSERT INTO "sqlite_sequence" VALUES('Director',2);
INSERT INTO "sqlite_sequence" VALUES('Role',2);
INSERT INTO "sqlite_sequence" VALUES('Theatre',2);
INSERT INTO "sqlite_sequence" VALUES('Performance',2);
INSERT INTO "sqlite_sequence" VALUES('Play',2);
COMMIT;
