BEGIN TRANSACTION;
CREATE TABLE "Action_Films" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    releaseYear INTEGER,
    genre TEXT,
    runtime INTEGER,
    language TEXT,
    film_festival_id INTEGER,
    FOREIGN KEY(film_festival_id) REFERENCES Festivals(id)
);
INSERT INTO "Action_Films" VALUES(1,'Inception',2010,'Action/Science Fiction',148,'English',1);
INSERT INTO "Action_Films" VALUES(2,'The Dark Knight',2008,'Action/Crime',152,'English',2);
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    dateOfBirth DATE,
    gender CHAR(1),
    ethnicity TEXT
);
INSERT INTO "Actors" VALUES(1,'Leonardo DiCaprio','1974-11-11','Male','Italian/German/Irish');
INSERT INTO "Actors" VALUES(2,'Jennifer Lawrence','1990-08-15','Female','England');
CREATE TABLE Blockbusters (
    id INTEGER PRIMARY KEY,
    title TEXT,
    releaseYear INTEGER,
    director TEXT,
    budget REAL,
    boxOffice REAL,
    film_festival_id INTEGER,
    FOREIGN KEY(film_festival_id) REFERENCES Festivals(id)
);
INSERT INTO "Blockbusters" VALUES(1,'Avatar',2009,'James Cameron',237.0,2747.0,1);
INSERT INTO "Blockbusters" VALUES(2,'Titanic',1997,'James Cameron',200.0,2202.0,2);
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthDate DATE,
    nationality TEXT
);
INSERT INTO "Directors" VALUES(1,'Quentin Tarantino','1963-03-27','American');
INSERT INTO "Directors" VALUES(2,'Christopher Nolan','1970-07-30','English');
CREATE TABLE Festivals (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Festivals" VALUES(1,'Sundance Film Festival');
INSERT INTO "Festivals" VALUES(2,'Cannes Film Festival');
CREATE TABLE Indies (
    id INTEGER PRIMARY KEY,
    title TEXT,
    releaseYear INTEGER,
    director TEXT,
    budget REAL,
    gross REAL,
    film_festival_id INTEGER,
    FOREIGN KEY(film_festival_id) REFERENCES Festivals(id)
);
INSERT INTO "Indies" VALUES(1,'Little Miss Sunshine',2006,'Jonathan Dayton',8.0,60.0,1);
INSERT INTO "Indies" VALUES(2,'Juno',2007,'Jason Reitman',6.0,231.0,2);
CREATE TABLE Roles (
    actor_id INTEGER,
    film_id INTEGER,
    characterName TEXT,
    FOREIGN KEY(actor_id) REFERENCES Actors(id),
    FOREIGN KEY(film_id) REFERENCES "Action_Films"(id)
);
INSERT INTO "Roles" VALUES(1,1,'Dom Cobb');
INSERT INTO "Roles" VALUES(2,1,'Rachel McDougall');
INSERT INTO "Roles" VALUES(1,2,'Howard St John');
INSERT INTO "Roles" VALUES(2,2,'Harvey Dent');
COMMIT;
