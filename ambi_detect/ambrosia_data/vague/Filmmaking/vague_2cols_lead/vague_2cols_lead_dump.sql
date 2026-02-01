BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY,
    DateOfBirth DATE
, fullname TEXT);
INSERT INTO "Actor" VALUES(1,'1974-11-11','Leonardo DiCaprio');
INSERT INTO "Actor" VALUES(2,'1964-09-02','Keanu Reeves');
INSERT INTO "Actor" VALUES(3,'1962-07-03','Tom Cruise');
INSERT INTO "Actor" VALUES(4,'1961-10-30','William Defoe');
INSERT INTO "Actor" VALUES(5,'1960-04-04','Hugo Weaving');
CREATE TABLE Actress(
    id INTEGER PRIMARY KEY,
    DateOfBirth DATE
, fullname TEXT);
INSERT INTO "Actress" VALUES(1,'1975-09-30','Marion Cotillard');
INSERT INTO "Actress" VALUES(2,'1987-11-28','Karen Gillan');
INSERT INTO "Actress" VALUES(3,'1976-07-15','Diane Kruger');
INSERT INTO "Actress" VALUES(4,'1987-02-21','Ellen Page');
INSERT INTO "Actress" VALUES(5,'1981-06-09','Natalie Portman');
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
CREATE TABLE Movie(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Genre TEXT,
    ReleaseDate DATE,
    DirectorId INTEGER,
    LeadActorId INTEGER,
    LeadActressId INTEGER,
    FOREIGN KEY (DirectorId) REFERENCES Director(id),
    FOREIGN KEY (LeadActorId) REFERENCES Actor(id),
    FOREIGN KEY (LeadActressId) REFERENCES Actress(id)
);
INSERT INTO "Movie" VALUES(1,'Inception','Action/Adventure','2010-07-16',1,3,4);
INSERT INTO "Movie" VALUES(2,'The Matrix','Science Fiction','1999-03-31',2,1,5);
INSERT INTO "Movie" VALUES(3,'Pulp Fiction','Crime/Drama','1994-10-14',3,2,6);
INSERT INTO "Movie" VALUES(4,'Titanic','Romance/Disaster','1997-12-18',4,1,7);
INSERT INTO "Movie" VALUES(5,'Interstellar','Science Fiction/Adventure','2014-11-06',5,3,8);
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
