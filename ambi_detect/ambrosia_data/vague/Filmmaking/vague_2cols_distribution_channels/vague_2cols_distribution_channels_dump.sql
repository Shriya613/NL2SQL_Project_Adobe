BEGIN TRANSACTION;
CREATE TABLE Actors (
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    DateOfBirth DATE
);
INSERT INTO "Actors" VALUES(1,'Leonardo DiCaprio','1974-11-11');
INSERT INTO "Actors" VALUES(2,'Tom Hanks','1956-07-09');
INSERT INTO "Actors" VALUES(3,'Scarlett Johansson','1984-11-22');
INSERT INTO "Actors" VALUES(4,'Robert Downey Jr.','1965-04-04');
INSERT INTO "Actors" VALUES(5,'Cate Blanchett','1969-05-14');
CREATE TABLE Directors (
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE
);
INSERT INTO "Directors" VALUES(1,'Martin Scorsese','1942-01-30');
INSERT INTO "Directors" VALUES(2,'Quentin Tarantino','1963-03-27');
INSERT INTO "Directors" VALUES(3,'Christopher Nolan','1970-07-30');
INSERT INTO "Directors" VALUES(4,'James Cameron','1954-8-16');
INSERT INTO "Directors" VALUES(5,'Steven Spielberg','1946-12-18');
CREATE TABLE Films (
    FilmID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INT,
    DirectorID INTEGER,
    ProducerID INTEGER,
    FOREIGN KEY(DirectorID) REFERENCES Directors(DirectorID),
    FOREIGN KEY(ProducerID) REFERENCES Producers(ProducerID)
);
INSERT INTO "Films" VALUES(1,'The Departed',2006,1,1);
INSERT INTO "Films" VALUES(2,'Pulp Fiction',1994,2,4);
INSERT INTO "Films" VALUES(3,'Inception',2010,3,6);
INSERT INTO "Films" VALUES(4,'Titanic',1997,4,1);
INSERT INTO "Films" VALUES(5,'Saving Private Ryan',1998,4,1);
CREATE TABLE Producers (
    ProducerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    ContactNumber BIGINT UNIQUE
);
INSERT INTO "Producers" VALUES(1,'Harvey Weinstein',1234567890);
INSERT INTO "Producers" VALUES(2,'David Fincher',1345678901);
INSERT INTO "Producers" VALUES(3,'Amy Pascal',1456789012);
INSERT INTO "Producers" VALUES(4,'Kathleen Kennedy',1567890134);
INSERT INTO "Producers" VALUES(5,'Frank Marshall',1678901456);
CREATE TABLE Roles (
    RoleID INTEGER PRIMARY KEY AUTOINCREMENT,
    FilmID INTEGER,
    ActorID INTEGER,
    CharacterName TEXT,
    FOREIGN KEY(FilmID) REFERENCES Films(FilmID),
    FOREIGN KEY(ActorID) REFERENCES Actors(ActorID)
);
INSERT INTO "Roles" VALUES(1,1,1,'William Costigan');
INSERT INTO "Roles" VALUES(2,2,2,'Vincent Vega');
INSERT INTO "Roles" VALUES(3,3,3,'Cobb');
INSERT INTO "Roles" VALUES(4,4,4,'Jack Dawson');
INSERT INTO "Roles" VALUES(5,4,5,'Captain Andrews');
CREATE TABLE StreamingDetails (
    DetailsID INTEGER PRIMARY KEY AUTOINCREMENT,
    Platforms TEXT,
    Countries TEXT,
    Movies TEXT,
    AvailableSince DATE,
    FOREIGN KEY(Movies) REFERENCES Films(Title)
);
INSERT INTO "StreamingDetails" VALUES(1,'Netflix','USA, UK','The Departed','2017-01-01');
INSERT INTO "StreamingDetails" VALUES(2,'Amazon Prime Video','Germany, France','Pulp Fiction','2018-02-01');
INSERT INTO "StreamingDetails" VALUES(3,'Hulu','Canada, Australia','Inception','2019-03-01');
INSERT INTO "StreamingDetails" VALUES(4,'Disney Plus','Worldwide','Titanic','2020-05-01');
INSERT INTO "StreamingDetails" VALUES(5,'Apple TV+','India, South Korea','Saving Private Ryan','2021-06-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Producers',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Films',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',5);
INSERT INTO "sqlite_sequence" VALUES('StreamingDetails',5);
COMMIT;
