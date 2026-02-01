BEGIN TRANSACTION;
CREATE TABLE Actors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Role TEXT);
INSERT INTO "Actors" VALUES(1,'Leonardo DiCaprio',45,'Dom Cobb');
INSERT INTO "Actors" VALUES(2,'Ellen Page',35,'Cole Tatum');
INSERT INTO "Actors" VALUES(3,'Tom Hardy',40,'Bane');
CREATE TABLE Directors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Movies_Directed INT,
    Director TEXT);
INSERT INTO "Directors" VALUES(1,'Quentin Tarantino',30,'Pulp Fiction');
INSERT INTO "Directors" VALUES(2,'James Cameron',40,'Avatar');
INSERT INTO "Directors" VALUES(3,'Steven Spielberg',50,'Indiana Jones');
CREATE TABLE Editors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Specialization TEXT,
    Experience INT,
    Editor TEXT,
    Deadline TEXT);
INSERT INTO "Editors" VALUES(1,'Michael Bay','Action',30,'Transformers','21.04.03');
INSERT INTO "Editors" VALUES(2,'Guillermo del Toro','Fantasy',25,'Pan''s Labyrinth','17.06.12');
INSERT INTO "Editors" VALUES(3,'Martin Scorsese','Drama',40,'Taxi Driver','15.05.01');
CREATE TABLE MovieCrew(
    ID INTEGER PRIMARY KEY,
    Position TEXT,
    PersonID INTEGER,
    HouseID INTEGER,
    FOREIGN KEY (PersonID) REFERENCES Actors(ID),
    FOREIGN KEY (HouseID) REFERENCES ProductionHouses(ID));
INSERT INTO "MovieCrew" VALUES(1,'Producer',1,1);
INSERT INTO "MovieCrew" VALUES(2,'Camera Operator',2,2);
INSERT INTO "MovieCrew" VALUES(3,'Sound Engineer',3,3);
INSERT INTO "MovieCrew" VALUES(4,'Set Designer',4,1);
INSERT INTO "MovieCrew" VALUES(5,'Costume Designer',5,1);
CREATE TABLE ProductionHouses(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT);
INSERT INTO "ProductionHouses" VALUES(1,'Warner Bros.','Los Angeles');
INSERT INTO "ProductionHouses" VALUES(2,'Paramount Pictures','Los Angeles');
INSERT INTO "ProductionHouses" VALUES(3,'Universal Studios','New York');
CREATE TABLE Scripts(
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Genre TEXT,
    Duration REAL,
    Scriptwriter TEXT,
    Deadline TEXT);
INSERT INTO "Scripts" VALUES(1,'Inception','Thriller',90.0,'Christopher Nolan','21.04.03');
INSERT INTO "Scripts" VALUES(2,'Interstellar','Science Fiction',168.0,'Christopher Nolan','20.07.01');
INSERT INTO "Scripts" VALUES(3,'The Dark Knight','Action',152.0,'Christopher Nolan','19.07.01');
COMMIT;
