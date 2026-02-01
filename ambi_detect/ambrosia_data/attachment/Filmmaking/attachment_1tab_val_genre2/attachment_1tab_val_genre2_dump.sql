BEGIN TRANSACTION;
CREATE TABLE Actors(
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    DateOfBirth DATE);
INSERT INTO "Actors" VALUES(1,'Chris Evans','1981-05-13');
INSERT INTO "Actors" VALUES(2,'Scarlett Johansson','1984-11-22');
INSERT INTO "Actors" VALUES(3,'Ryan Reynolds','1976-10-23');
INSERT INTO "Actors" VALUES(4,'Anne Hathaway','1982-11-12');
INSERT INTO "Actors" VALUES(5,'Christopher Nolan','1970-07-30');
CREATE TABLE Casts(
    FilmID INTEGER,
    ActorID INTEGER,
    Role TEXT,
    FOREIGN KEY (ActorID) REFERENCES Actors(ActorID),
    FOREIGN KEY (FilmID) REFERENCES Films(FilmID));
INSERT INTO "Casts" VALUES(1,1,'Leading Man');
INSERT INTO "Casts" VALUES(1,2,'Supporting Actor');
INSERT INTO "Casts" VALUES(2,3,'Funny Sidekick');
INSERT INTO "Casts" VALUES(2,4,'Serious Best Friend');
INSERT INTO "Casts" VALUES(3,5,'Wise Mentor');
CREATE TABLE Directors(
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE);
INSERT INTO "Directors" VALUES(1,'John Doe','1980-07-30');
INSERT INTO "Directors" VALUES(2,'Jane Smith','1965-01-15');
INSERT INTO "Directors" VALUES(3,'Michael Johnson','1992-11-23');
INSERT INTO "Directors" VALUES(4,'Emily Davis','1978-05-28');
INSERT INTO "Directors" VALUES(5,'William Brown','1950-12-04');
CREATE TABLE Films(
    FilmID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Genre TEXT,
    Release_Year INT ,
    DirectorID INTEGER,
    ProducerID INTEGER,
    FOREIGN KEY (DirectorID) REFERENCES Directors(DirectorID),
    FOREIGN KEY (ProducerID) REFERENCES Producers(ProducerID));
INSERT INTO "Films" VALUES(1,'Animated Fun','Animation',2021,1,1);
INSERT INTO "Films" VALUES(2,'A Comic Adventure','Comedy',2021,2,2);
INSERT INTO "Films" VALUES(3,'The Silent Clown','Comedy',1999,3,3);
INSERT INTO "Films" VALUES(4,'Fantastic Journey','Animation',2019,4,4);
INSERT INTO "Films" VALUES(5,'Dream Chasers','Drama',2005,5,5);
CREATE TABLE Producers(
    ProducerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    ProductionCompany TEXT);
INSERT INTO "Producers" VALUES(1,'Alice Green','Green Studios');
INSERT INTO "Producers" VALUES(2,'Robert White','White Pictures');
INSERT INTO "Producers" VALUES(3,'David Black','Black Entertainment');
INSERT INTO "Producers" VALUES(4,'Carol Blue','Blue Vision');
INSERT INTO "Producers" VALUES(5,'Peter Gray','Gray Media');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Producers',5);
INSERT INTO "sqlite_sequence" VALUES('Films',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
COMMIT;
