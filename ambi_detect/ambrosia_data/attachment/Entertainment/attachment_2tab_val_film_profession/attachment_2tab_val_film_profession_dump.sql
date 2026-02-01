BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT
);
INSERT INTO "Actor" VALUES(1,'Tom Hanks','1956-07-09','USA');
INSERT INTO "Actor" VALUES(2,'Helen Mirren','1945-07-26','UK');
CREATE TABLE Casting (
    movie_id INTEGER,
    actor_id INTEGER,
    role TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movie(id),
    FOREIGN KEY(actor_id) REFERENCES Actor(id)
);
INSERT INTO "Casting" VALUES(1,1,'Lead Role');
INSERT INTO "Casting" VALUES(1,2,'Supporting Role');
CREATE TABLE Director (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT,
    Award TEXT
);
INSERT INTO "Director" VALUES(1,'John Doe','1960-07-28','USA','Oscar');
INSERT INTO "Director" VALUES(2,'Jane Smith','1970-03-12','Canada','Golden Globe');
CREATE TABLE Movie (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INT,
    Genre TEXT,
    Rating FLOAT
);
INSERT INTO "Movie" VALUES(1,'Movie1',2000,'Drama',8.5);
INSERT INTO "Movie" VALUES(2,'Movie2',2005,'Comedy',7.0);
CREATE TABLE Producer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT,
    Award TEXT
);
INSERT INTO "Producer" VALUES(1,'Alice Johnson','1955-09-03','UK','Oscar');
INSERT INTO "Producer" VALUES(2,'Bob Brown','1965-12-15','Australia','BAFTA');
CREATE TABLE Review (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id INTEGER,
    rating INT,
    comment TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movie(id)
);
INSERT INTO "Review" VALUES(1,1,9,'Great film! Loved it.');
INSERT INTO "Review" VALUES(2,2,8,'Good movie but could have been better.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Director',2);
INSERT INTO "sqlite_sequence" VALUES('Producer',2);
INSERT INTO "sqlite_sequence" VALUES('Movie',2);
INSERT INTO "sqlite_sequence" VALUES('Actor',2);
INSERT INTO "sqlite_sequence" VALUES('Review',2);
COMMIT;
