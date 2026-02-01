BEGIN TRANSACTION;
CREATE TABLE Actors(
    Actor_ID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Height DOUBLE );
INSERT INTO "Actors" VALUES(1,'Chris Johnson','Johnson','1990-01-10',185.3);
INSERT INTO "Actors" VALUES(2,'Emma Williams','Williams','1992-08-05',168.2);
CREATE TABLE Crew(
    Crew_ID INTEGER PRIMARY KEY,
    Position TEXT,
    Name TEXT,
    Experience TEXT,
    Episode_ID INTEGER,
    FOREIGN KEY(Episode_ID) REFERENCES Episodes(Episode_ID));
INSERT INTO "Crew" VALUES(1,'Director','David Miller','10 years',1);
INSERT INTO "Crew" VALUES(2,'Editor','Samantha Taylor','8 years',1);
INSERT INTO "Crew" VALUES(3,'Director','Lisa Thompson','7 years',2);
INSERT INTO "Crew" VALUES(4,'Editor','Robert Harris','6 years',2);
CREATE TABLE Episodes(
    Episode_ID INTEGER PRIMARY KEY,
    SeasonNumber INTEGER ,
    EpisodeTitle TEXT,
    EpisodeLength TIME,
    AirDate DATE,
    Crew_ID INTEGER,
    FOREIGN KEY(Crew_ID) REFERENCES Crews(Crew_ID));
INSERT INTO "Episodes" VALUES(1,1,'Pilot Episode','01:00:00','2021-09-15',1);
INSERT INTO "Episodes" VALUES(2,1,'Second Episode','00:45:00','2021-09-22',2);
CREATE TABLE "Movies"(
    Movie_ID INTEGER PRIMARY KEY,
    Title TEXT,
    ReleaseDate DATE,
    Genre TEXT,
    BoxOffice REAL);
INSERT INTO "Movies" VALUES(1,'Action Movie','2020-10-01','Action',100000000.0);
INSERT INTO "Movies" VALUES(2,'Drama Film','2019-05-15','Drama',80000000.0);
CREATE TABLE Roles(
    Role_ID INTEGER PRIMARY KEY,
    CharacterName TEXT,
    Movie_ID INTEGER,
    Actor_ID INTEGER,
    FOREIGN KEY(Movie_ID) REFERENCES Movies(Movie_ID),
    FOREIGN KEY(Actor_ID) REFERENCES Actors(Actor_ID));
INSERT INTO "Roles" VALUES(1,'Hero',1,1);
INSERT INTO "Roles" VALUES(2,'Villain',2,2);
COMMIT;
