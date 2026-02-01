BEGIN TRANSACTION;
CREATE TABLE Actors(
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    DateOfBirth DATE,
    Nationality TEXT);
INSERT INTO "Actors" VALUES(1,'Leonardo DiCaprio','1974-11-11','United States');
INSERT INTO "Actors" VALUES(2,'Meryl Streep','1949-06-22','United States');
INSERT INTO "Actors" VALUES(3,'Jennifer Lawrence','1990-08-15','United States');
INSERT INTO "Actors" VALUES(4,'Robert Downey Jr.','1965-04-04','United States');
INSERT INTO "Actors" VALUES(5,'Gal Gadot','1985-04-30','Israel');
CREATE TABLE Directors(
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT);
INSERT INTO "Directors" VALUES(1,'Christopher Nolan','1970-07-30','United Kingdom');
INSERT INTO "Directors" VALUES(2,'Quentin Tarantino','1963-03-27','United States');
INSERT INTO "Directors" VALUES(3,'Martin Scorsese','1942-11-22','United States');
INSERT INTO "Directors" VALUES(4,'Ava DuVernay','1982-08-24','United States');
INSERT INTO "Directors" VALUES(5,'Denis Villeneuve','1967-03-03','Canada');
CREATE TABLE Movies(
    MovieID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INT,
    Genre TEXT,
    Rating TEXT,
    FilmFestival TEXT,
    Budget INT,
    BoxOffice INT,
    DirectorID INTEGER,
    ProducerID INTEGER,
    FOREIGN KEY(DirectorID) REFERENCES Directors(DirectorID),
    FOREIGN KEY(ProducerID) REFERENCES Producers(ProducerID));
INSERT INTO "Movies" VALUES(1,'Inception',2010,'Blockbuster','PG-13','Sundance Film Festival',160,829,1,1);
INSERT INTO "Movies" VALUES(2,'Titanic',1997,'Blockbuster','PG-13','Academy Awards',200,2202,NULL,2);
INSERT INTO "Movies" VALUES(3,'Good Will Hunting',1997,'Indie Film','R','Sundance Film Festival',30,330,NULL,3);
INSERT INTO "Movies" VALUES(4,'Little Miss Sunshine',2006,'Indie Film','PG-13','Sundance Film Festival',8,60,NULL,4);
INSERT INTO "Movies" VALUES(5,'Dunkirk',2017,'Blockbuster','PG-13','Cannes Film Festival',100,527,1,1);
CREATE TABLE Producers(
    ProducerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    CompanyName TEXT,
    ContactInfo TEXT);
INSERT INTO "Producers" VALUES(1,'Kevin Feige','Marvel Studios','kevin@marvelstudios.com');
INSERT INTO "Producers" VALUES(2,'Frank Marshall','Amblin Entertainment','frank@amblinentertainment.com');
INSERT INTO "Producers" VALUES(3,'Barbara Broccoli','EON Productions','barbara@eonproductions.com');
INSERT INTO "Producers" VALUES(4,'David Heyman','Heyday Films','david@heydayfilms.com');
INSERT INTO "Producers" VALUES(5,'Jason Blum','Blumhouse Productions','jason@blumhouse.com');
CREATE TABLE Roles(
    RoleID INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieTitle TEXT,
    CharacterName TEXT,
    DirectorID INTEGER,
    ProducerID INTEGER,
    FOREIGN KEY(DirectorID) REFERENCES Directors(DirectorID),
    FOREIGN KEY(ProducerID) REFERENCES Producers(ProducerID));
INSERT INTO "Roles" VALUES(1,'Inception','Cobb',1,1);
INSERT INTO "Roles" VALUES(2,'Pulp Fiction','Mia Wallace',2,2);
INSERT INTO "Roles" VALUES(3,'The Departed','Colin Sullivan',3,3);
INSERT INTO "Roles" VALUES(4,'A Wrinkle in Time','Meg Murry',4,4);
INSERT INTO "Roles" VALUES(5,'Interstellar','Cooper',1,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Producers',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
COMMIT;
