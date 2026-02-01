BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date_of_birth DATE
);
INSERT INTO "Actor" VALUES(1,'Tom Hanks','1956-07-09');
INSERT INTO "Actor" VALUES(2,'Meryl Streep','1949-06-22');
CREATE TABLE "Documentary" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    producer_id INTEGER,
    FOREIGN KEY (producer_id) REFERENCES "Production"(id));
INSERT INTO "Documentary" VALUES(1,'The Art of Filmmaking','2020-12-31',1);
INSERT INTO "Documentary" VALUES(2,'Behind the Scenes','2021-05-01',2);
CREATE TABLE Movie (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    budget DOUBLE,
    box_office DOUBLE,
    foreign_key INTEGER,
    FOREIGN KEY (foreign_key) REFERENCES Documentary(id)
);
INSERT INTO "Movie" VALUES(1,'Inception','2010-07-16',160000000.0,829000000.0,1);
INSERT INTO "Movie" VALUES(2,'Avatar','2009-12-18',237000000.0,2743460000.0,2);
CREATE TABLE "Production" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_date DATE
);
INSERT INTO "Production" VALUES(1,'John Doe','1970-01-01');
INSERT INTO "Production" VALUES(2,'Jane Smith','1980-06-23');
CREATE TABLE Review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rating INTEGER ,
    reviewer_name TEXT,
    comment TEXT,
    movie_id INTEGER,
    documentary_id INTEGER,
    FOREIGN KEY (movie_id) REFERENCES Movie(id),
    FOREIGN KEY (documentary_id) REFERENCES Documentary(id)
);
INSERT INTO "Review" VALUES(1,9,'Alice Johnson','Great movie with a captivating storyline.',1,NULL);
INSERT INTO "Review" VALUES(2,8,'Bob Brown','A must-watch for any film enthusiast.',2,NULL);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Production',2);
INSERT INTO "sqlite_sequence" VALUES('Actor',2);
INSERT INTO "sqlite_sequence" VALUES('Movie',2);
INSERT INTO "sqlite_sequence" VALUES('Review',2);
INSERT INTO "sqlite_sequence" VALUES('Documentary',2);
COMMIT;
