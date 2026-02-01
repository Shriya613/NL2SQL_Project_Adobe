BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Actor" VALUES(1,'Leonardo','DiCaprio','1974-11-11');
INSERT INTO "Actor" VALUES(2,'Keanu','Reeves','1964-09-02');
INSERT INTO "Actor" VALUES(3,'Tom','Cruise','1962-07-03');
INSERT INTO "Actor" VALUES(4,'William','Defoe','1961-10-30');
INSERT INTO "Actor" VALUES(5,'Hugo','Weaving','1960-04-04');
CREATE TABLE Actress(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Actress" VALUES(1,'Marion','Cotillard','1975-09-30');
INSERT INTO "Actress" VALUES(2,'Karen','Gillan','1987-11-28');
INSERT INTO "Actress" VALUES(3,'Diane','Kruger','1976-07-15');
INSERT INTO "Actress" VALUES(4,'Ellen','Page','1987-02-21');
INSERT INTO "Actress" VALUES(5,'Natalie','Portman','1981-06-09');
CREATE TABLE Director(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Birthdate DATE
);
INSERT INTO "Director" VALUES(1,'Christopher Nolan','1970-07-30');
INSERT INTO "Director" VALUES(2,'Lana Wachowski','1969-06-21');
INSERT INTO "Director" VALUES(3,'Quentin Tarantino','1963-03-27');
INSERT INTO "Director" VALUES(4,'James Cameron','1954-08-16');
INSERT INTO "Director" VALUES(5,'Christopher McQuarrie','1968-06-04');
CREATE TABLE "Movie"(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Genre TEXT,
    ReleaseDate DATE,
    DirectorId INTEGER,
    Lead INTEGER,
    FOREIGN KEY (DirectorId) REFERENCES Director(id),
    FOREIGN KEY (Lead) REFERENCES Actor(id));
INSERT INTO "Movie" VALUES(1,'Inception','Action/Adventure','2010-07-16',1,3);
INSERT INTO "Movie" VALUES(2,'The Matrix','Science Fiction','1999-03-31',2,1);
INSERT INTO "Movie" VALUES(3,'Pulp Fiction','Crime/Drama','1994-10-14',3,2);
INSERT INTO "Movie" VALUES(4,'Titanic','Romance/Disaster','1997-12-18',4,1);
INSERT INTO "Movie" VALUES(5,'Interstellar','Science Fiction/Adventure','2014-11-06',5,3);
CREATE TABLE Role(
    id INTEGER PRIMARY KEY,
    CharacterName TEXT,
    MovieId INTEGER,
    ActorId INTEGER,
    ActressId INTEGER,
    FOREIGN KEY (MovieId) REFERENCES Movie(id),
    FOREIGN KEY (ActorId) REFERENCES Actor(id),
    FOREIGN KEY (ActressId) REFERENCES Actress(id)
);
INSERT INTO "Role" VALUES(1,'Dom Cobb',1,2,4);
INSERT INTO "Role" VALUES(2,'Neo',2,1,5);
INSERT INTO "Role" VALUES(3,'Floyd Collins',3,3,6);
INSERT INTO "Role" VALUES(4,'Rose DeWitt Bukater',4,1,7);
INSERT INTO "Role" VALUES(5,'Cooper',5,3,8);
COMMIT;
