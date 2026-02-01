BEGIN TRANSACTION;
CREATE TABLE Author(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Author" VALUES(1,'John','Doe');
INSERT INTO "Author" VALUES(2,'Jane','Smith');
CREATE TABLE EntertainmentItem(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseYear INT,
    genreId INTEGER,
    mediaTypeId INTEGER,
    FOREIGN KEY (genreId) REFERENCES Genre(id),
    FOREIGN KEY (mediaTypeId) REFERENCES MediaType(id)
);
INSERT INTO "EntertainmentItem" VALUES(1,'Film1',2020,1,1);
INSERT INTO "EntertainmentItem" VALUES(2,'Film2',2021,2,1);
INSERT INTO "EntertainmentItem" VALUES(3,'Novel1',2020,3,2);
INSERT INTO "EntertainmentItem" VALUES(4,'Novel2',2021,4,2);
CREATE TABLE Film(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    director TEXT,
    filmId INTEGER,
    FOREIGN KEY (filmId) REFERENCES EntertainmentItem(id)
);
INSERT INTO "Film" VALUES(1,'Director1',1);
INSERT INTO "Film" VALUES(2,'Director2',2);
CREATE TABLE Genre(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Action');
INSERT INTO "Genre" VALUES(2,'Adventure');
INSERT INTO "Genre" VALUES(3,'Comedy');
INSERT INTO "Genre" VALUES(4,'Drama');
INSERT INTO "Genre" VALUES(5,'Fantasy');
CREATE TABLE MediaType(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    typeName TEXT UNIQUE
);
INSERT INTO "MediaType" VALUES(1,'Film');
INSERT INTO "MediaType" VALUES(2,'Novel');
CREATE TABLE Novel(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    authorId INTEGER,
    publicationYear INT,
    FOREIGN KEY (authorId) REFERENCES Author(id)
);
INSERT INTO "Novel" VALUES(1,1,2020);
INSERT INTO "Novel" VALUES(2,2,2021);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Genre',5);
INSERT INTO "sqlite_sequence" VALUES('MediaType',2);
INSERT INTO "sqlite_sequence" VALUES('EntertainmentItem',4);
INSERT INTO "sqlite_sequence" VALUES('Author',2);
INSERT INTO "sqlite_sequence" VALUES('Novel',2);
INSERT INTO "sqlite_sequence" VALUES('Film',2);
COMMIT;
