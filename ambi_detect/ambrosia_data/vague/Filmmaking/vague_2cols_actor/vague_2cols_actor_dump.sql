BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "fullName" TEXT,
    birthDate DATE);
INSERT INTO "Actor" VALUES(1,'Leonardo DiCaprio','1974-11-11');
INSERT INTO "Actor" VALUES(2,'Matthew McConaughey','1969-11-04');
INSERT INTO "Actor" VALUES(3,'Brad Pitt','1963-12-18');
INSERT INTO "Actor" VALUES(4,'Angelina Jolie','1975-06-04');
INSERT INTO "Actor" VALUES(5,'Cate Blanchett','1969-05-14');
INSERT INTO "Actor" VALUES(6,'Tom Hardy',NULL);
CREATE TABLE Director(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthDate DATE);
INSERT INTO "Director" VALUES(1,'Quentin Tarantino','1963-03-27');
INSERT INTO "Director" VALUES(2,'Christopher Nolan','1970-07-30');
INSERT INTO "Director" VALUES(3,'Martin Scorsese','1942-11-22');
INSERT INTO "Director" VALUES(4,'Ava DuVernay','1978-08-24');
INSERT INTO "Director" VALUES(5,'James Cameron','1954-08-16');
CREATE TABLE Genre(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Genre" VALUES(1,'Drama');
INSERT INTO "Genre" VALUES(2,'Thriller');
INSERT INTO "Genre" VALUES(3,'Adventure');
INSERT INTO "Genre" VALUES(4,'Horror');
INSERT INTO "Genre" VALUES(5,'Romance');
CREATE TABLE Movie(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseYear INTEGER,
    directorId INTEGER,
    FOREIGN KEY(directorId) REFERENCES Director(id));
INSERT INTO "Movie" VALUES(1,'Inception',2010,1);
INSERT INTO "Movie" VALUES(2,'Interstellar',2014,2);
INSERT INTO "Movie" VALUES(3,'The Wolf of Wall Street',2013,1);
INSERT INTO "Movie" VALUES(4,'Avatar',2009,3);
INSERT INTO "Movie" VALUES(5,'Black Swan',2010,4);
CREATE TABLE Rating(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roleId INTEGER,
    score INTEGER ,
    FOREIGN KEY(roleId) REFERENCES Role(id));
INSERT INTO "Rating" VALUES(1,1,9);
INSERT INTO "Rating" VALUES(2,2,8);
INSERT INTO "Rating" VALUES(3,3,7);
INSERT INTO "Rating" VALUES(4,4,9);
INSERT INTO "Rating" VALUES(5,5,8);
CREATE TABLE Role(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movieId INTEGER,
    leadRole BOOLEAN,
    supportingRole BOOLEAN,
    actorId INTEGER,
    genreId INTEGER,
    rating INTEGER,
    FOREIGN KEY(movieId) REFERENCES Movie(id),
    FOREIGN KEY(actorId) REFERENCES Actor(id),
    FOREIGN KEY(genreId) REFERENCES Genre(id));
INSERT INTO "Role" VALUES(1,1,1,0,1,1,9);
INSERT INTO "Role" VALUES(2,1,0,1,2,2,8);
INSERT INTO "Role" VALUES(3,2,0,1,3,3,7);
INSERT INTO "Role" VALUES(4,3,1,0,4,4,9);
INSERT INTO "Role" VALUES(5,3,0,1,5,5,8);
INSERT INTO "Role" VALUES(6,1,0,1,6,1,9);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',6);
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
INSERT INTO "sqlite_sequence" VALUES('Role',6);
INSERT INTO "sqlite_sequence" VALUES('Genre',5);
INSERT INTO "sqlite_sequence" VALUES('Rating',5);
COMMIT;
