BEGIN TRANSACTION;
CREATE TABLE "Composer"( `id` INTEGER PRIMARY KEY ,
    creator TEXT,
    birth_year INTEGER,
    death_year INTEGER
);
INSERT INTO "Composer" VALUES(1,'John Williams',1932,NULL);
INSERT INTO "Composer" VALUES(2,'Hans Zimmer',1957,NULL);
INSERT INTO "Composer" VALUES(3,'Danny Elfman',1953,NULL);
INSERT INTO "Composer" VALUES(4,'Howard Shore',1946,NULL);
INSERT INTO "Composer" VALUES(5,'Thomas Newman',1955,NULL);
INSERT INTO "Composer" VALUES(6,'Zack Hemsey',1983,NULL);
INSERT INTO "Composer" VALUES(7,'Javier Navarrete',1956,NULL);
CREATE TABLE Director(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Director" VALUES(1,'Christopher Nolan');
INSERT INTO "Director" VALUES(2,'Quentin Tarantino');
INSERT INTO "Director" VALUES(3,'Steven Spielberg');
INSERT INTO "Director" VALUES(4,'Guillermo del Toro');
INSERT INTO "Director" VALUES(5,'Martin Scorsese');
INSERT INTO "Director" VALUES(6,'Ava DuVernay');
CREATE TABLE Film(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    director_id INTEGER,
    FOREIGN KEY (director_id) REFERENCES Director(id)
);
INSERT INTO "Film" VALUES(1,'Inception','2010-07-16',1);
INSERT INTO "Film" VALUES(2,'Pulp Fiction','1994-10-14',2);
INSERT INTO "Film" VALUES(3,'Schindler’s List','1993-12-15',3);
INSERT INTO "Film" VALUES(4,'Pan’s Labyrinth','2006-03-17',4);
INSERT INTO "Film" VALUES(5,'The Departed','2006-08-04',5);
CREATE TABLE "Movie_Elements"(
    element_type TEXT ,
    movie_title TEXT,
    "composer_id" INTEGER,
    duration INTEGER,
    FOREIGN KEY ("composer_id") REFERENCES Composer(id)
);
INSERT INTO "Movie_Elements" VALUES('Sound Design','Inception',6,140);
INSERT INTO "Movie_Elements" VALUES('Sound Design','Inception',1,150);
INSERT INTO "Movie_Elements" VALUES('Music Score','Schindler’s List',1,150);
INSERT INTO "Movie_Elements" VALUES('Music Score','Pan’s Labyrinth',7,120);
INSERT INTO "Movie_Elements" VALUES('Music Score','The Departed',4,130);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Director',6);
INSERT INTO "sqlite_sequence" VALUES('Film',5);
COMMIT;
