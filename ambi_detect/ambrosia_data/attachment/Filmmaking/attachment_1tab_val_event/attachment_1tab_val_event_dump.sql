BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    director_id INTEGER,
    FOREIGN KEY(director_id) REFERENCES Director(id));
INSERT INTO "Actor" VALUES(1,'Tim Robbins',1);
INSERT INTO "Actor" VALUES(2,'Tom Hanks',1);
INSERT INTO "Actor" VALUES(3,'Liam Neeson',2);
INSERT INTO "Actor" VALUES(4,'Ben Kingsley',2);
INSERT INTO "Actor" VALUES(5,'Heath Ledger',3);
CREATE TABLE Character(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    actor_id INTEGER,
    movie_id INTEGER,
    FOREIGN KEY(actor_id) REFERENCES Actor(id),
    FOREIGN KEY(movie_id) REFERENCES Movie(id));
INSERT INTO "Character" VALUES(1,'Andy Dufresne',1,1);
INSERT INTO "Character" VALUES(2,'Randall Stephens',2,1);
INSERT INTO "Character" VALUES(3,'Maximilliano Oliveira',3,2);
INSERT INTO "Character" VALUES(4,'Janusz Bielski',4,3);
INSERT INTO "Character" VALUES(5,'Joker',5,4);
CREATE TABLE Director(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Director" VALUES(1,'Frank Darabont');
INSERT INTO "Director" VALUES(2,'Steven Spielberg');
INSERT INTO "Director" VALUES(3,'Francis Ford Coppola');
INSERT INTO "Director" VALUES(4,'Milos Forman');
INSERT INTO "Director" VALUES(5,'Christopher Nolan');
CREATE TABLE Festival(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    event TEXT,
    committee_members TEXT);
INSERT INTO "Festival" VALUES(1,'Cannes Film Festival','Film Festival','Gerardo Prat');
INSERT INTO "Festival" VALUES(2,'Venice Film Festival','Film Festival','Maria Grazia Cucinotta');
INSERT INTO "Festival" VALUES(3,'Berlin International Film Festival','Film Festival','Christian Petzold');
INSERT INTO "Festival" VALUES(4,'Toronto International Film Festival','Film Festival','Piers Handling');
INSERT INTO "Festival" VALUES(5,'Academy Awards','Award','Gerardo Prat');
INSERT INTO "Festival" VALUES(6,'BAFTA Awards','Award','Amanda Berry');
CREATE TABLE Movie(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    festival_id INTEGER,
    FOREIGN KEY(festival_id) REFERENCES Festival(id));
INSERT INTO "Movie" VALUES(1,'The Shawshank Redemption',1994,1);
INSERT INTO "Movie" VALUES(2,'Schindler''s List',1993,2);
INSERT INTO "Movie" VALUES(3,'The Godfather',1972,3);
INSERT INTO "Movie" VALUES(4,'One Flew Over the Cuckoo''s Nest',1975,4);
INSERT INTO "Movie" VALUES(5,'The Dark Knight',2008,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Festival',6);
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Character',5);
COMMIT;
