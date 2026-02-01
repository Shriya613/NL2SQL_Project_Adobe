BEGIN TRANSACTION;
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY,
    firstname TEXT,
    lastname TEXT,
    birthdate DATE,
    roleInFilms TEXT
);
INSERT INTO "Actors" VALUES(1,'Leonardo','DiCaprio','1974-11-11','Lead Actor');
INSERT INTO "Actors" VALUES(2,'Tom','Hardy','1977-09-15','Supporting Actor');
INSERT INTO "Actors" VALUES(3,'Joseph','Gordon-Levitt','1981-02-17','Supporting Actor');
INSERT INTO "Actors" VALUES(4,'Cate','Blanchett','1969-05-14','Lead Actress');
INSERT INTO "Actors" VALUES(5,'Marion','Cotillard','1975-09-30','Supporting Actress');
CREATE TABLE Awards (
    id INTEGER PRIMARY KEY,
    awardTitle TEXT,
    filmId INTEGER,
    yearOfAward INT,
    FOREIGN KEY(filmId) REFERENCES Films(id)
);
INSERT INTO "Awards" VALUES(1,'Best Director',1,2011);
INSERT INTO "Awards" VALUES(2,'Best Supporting Actor',1,2011);
INSERT INTO "Awards" VALUES(3,'Best Original Screenplay',1,2011);
INSERT INTO "Awards" VALUES(4,'Best Picture',1,2011);
INSERT INTO "Awards" VALUES(5,'Best Cinematography',1,2011);
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    birthdate DATE,
    nationality TEXT
);
INSERT INTO "Directors" VALUES(1,'Christopher Nolan','1970-07-30','British');
INSERT INTO "Directors" VALUES(2,'Quentin Tarantino','1963-03-27','American');
INSERT INTO "Directors" VALUES(3,'Martin Scorsese','1942-11-22','American');
INSERT INTO "Directors" VALUES(4,'Steven Spielberg','1946-12-18','American');
INSERT INTO "Directors" VALUES(5,'Guillermo del Toro','1964-10-09','Mexican');
CREATE TABLE "Films" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    releaseyear INT,
    Budget REAL,
    genre TEXT,
    directorId INTEGER,
    producerId INTEGER,
    FOREIGN KEY(directorId) REFERENCES Directors(id),
    FOREIGN KEY(producerId) REFERENCES Producers(id)
);
INSERT INTO "Films" VALUES(1,'Inception',2010,160000000.0,'Action/Science Fiction',1,1);
INSERT INTO "Films" VALUES(2,'Pulp Fiction',1994,8000000.0,'Crime/Drama',2,3);
INSERT INTO "Films" VALUES(3,'Taxi Driver',1976,2000000.0,'Thriller/Drama',4,5);
INSERT INTO "Films" VALUES(4,'Schindler''s List',1993,22000000.0,'Historical/War',3,2);
INSERT INTO "Films" VALUES(5,'The Social Network',2010,40000000.0,'Biographical/Drama',5,1);
CREATE TABLE Producers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    company TEXT,
    networth REAL
);
INSERT INTO "Producers" VALUES(1,'David Heyman','Heyday Films',NULL);
INSERT INTO "Producers" VALUES(2,'Jeremy Kleiner','Anonymous Content',NULL);
INSERT INTO "Producers" VALUES(3,'Dede Gardner','Plan B Entertainment',NULL);
INSERT INTO "Producers" VALUES(4,'Barbara Broccoli','EON Productions',NULL);
INSERT INTO "Producers" VALUES(5,'Frank Marshall','Amblin Partners',NULL);
CREATE TABLE Roles (
    id INTEGER PRIMARY KEY,
    actorId INTEGER,
    characterName TEXT,
    salary REAL,
    filmId INTEGER,
    FOREIGN KEY(actorId) REFERENCES Actors(id),
    FOREIGN KEY(filmId) REFERENCES Films(id)
);
INSERT INTO "Roles" VALUES(1,1,'Dom Cobb',10000000.0,1);
INSERT INTO "Roles" VALUES(2,2,'Eames',5000000.0,1);
INSERT INTO "Roles" VALUES(3,3,'Edward Alderson',4000000.0,1);
INSERT INTO "Roles" VALUES(4,4,'Elise',3000000.0,1);
INSERT INTO "Roles" VALUES(5,5,'Arthur',2000000.0,1);
COMMIT;
