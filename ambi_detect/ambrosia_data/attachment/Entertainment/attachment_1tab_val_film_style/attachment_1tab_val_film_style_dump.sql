BEGIN TRANSACTION;
CREATE TABLE Actor (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Birthdate DATE,
    Nationality TEXT,
    Gender TEXT
);
INSERT INTO "Actor" VALUES(1,'Tom Hanks','1956-07-09','American','Male');
INSERT INTO "Actor" VALUES(2,'Halle Berry','1966-08-14','American','Female');
CREATE TABLE Character (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Description TEXT,
    Role INTEGER,
    Film_ID INTEGER,
    FOREIGN KEY(Role) REFERENCES Actor(ID),
    FOREIGN KEY(Film_ID) REFERENCES Film(ID)
);
INSERT INTO "Character" VALUES(1,'Captain Nemo','A brave adventurer',1,1);
INSERT INTO "Character" VALUES(2,'Dr. John Dolittle','An animal communicator',2,2);
CREATE TABLE Film (
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    ReleaseYear INTEGER,
    Runtime INTEGER,
    Film_Style TEXT,
    Director TEXT,
    FOREIGN KEY(Director) REFERENCES Person(Name)
);
INSERT INTO "Film" VALUES(1,'The Adventures of Tom Sawyer',1986,90,'Adventure Movie','Sofia Martinez');
INSERT INTO "Film" VALUES(2,'Jurassic Park',2013,127,'Adventure Movie','Steven Spielberg');
INSERT INTO "Film" VALUES(3,'Inception',2021,120,'Action Movie','Sofia Martinez');
INSERT INTO "Film" VALUES(4,'Die Hard',2022,132,'Action Movie','John McTiernan');
CREATE TABLE FilmCrew (
    ID INTEGER PRIMARY KEY,
    Position TEXT,
    Department TEXT,
    Film_ID INTEGER,
    FOREIGN KEY(Film_ID) REFERENCES Film(ID)
);
INSERT INTO "FilmCrew" VALUES(1,'Director','Direction',1);
INSERT INTO "FilmCrew" VALUES(2,'Screenwriter','Script',2);
CREATE TABLE Producer (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Birthdate DATE,
    Nationality TEXT,
    ProductionCompany_ID INTEGER,
    FOREIGN KEY(ProductionCompany_ID) REFERENCES ProductionCompany(ID)
);
INSERT INTO "Producer" VALUES(1,'David Foster','1949-01-11','Canadian',1);
INSERT INTO "Producer" VALUES(2,'Karen Lunder','1978-01-15','Swedish',2);
CREATE TABLE ProductionCompany (
    ID INTEGER PRIMARY KEY,
    CompanyName TEXT,
    Address TEXT,
    ContactEmail TEXT,
    Founded DATE
);
INSERT INTO "ProductionCompany" VALUES(1,'Walt Disney Pictures','Burbank, CA','contact@disney.com','1923-10-16');
INSERT INTO "ProductionCompany" VALUES(2,'Universal Studios','Universal City, CA','info@universalstudios.com','1912-04-21');
COMMIT;
