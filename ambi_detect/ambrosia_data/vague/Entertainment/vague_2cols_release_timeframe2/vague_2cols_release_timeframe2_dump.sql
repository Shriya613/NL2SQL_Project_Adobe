BEGIN TRANSACTION;
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FullName VARCHAR(255),
    DateOfBirth DATE
);
INSERT INTO "Actors" VALUES(1,'Keanu Reeves','1964-09-02');
INSERT INTO "Actors" VALUES(2,'Kate Winslet','1975-10-05');
INSERT INTO "Actors" VALUES(3,'Chris Hemsworth','1983-08-11');
INSERT INTO "Actors" VALUES(4,'Scarlett Johansson','1984-11-22');
INSERT INTO "Actors" VALUES(5,'Christopher Nolan','1970-07-30');
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName VARCHAR(100),
    LastName VARCHAR(100),
    BirthDate DATE
);
INSERT INTO "Directors" VALUES(1,'John','Doe','1970-06-15');
INSERT INTO "Directors" VALUES(2,'Jane','Smith','1985-03-22');
INSERT INTO "Directors" VALUES(3,'Tom','Hanks','1956-07-09');
INSERT INTO "Directors" VALUES(4,'Emma','Watson','1990-04-15');
INSERT INTO "Directors" VALUES(5,'Leonardo','DiCaprio','1974-11-11');
CREATE TABLE GameReviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ReviewText TEXT,
    Rating INT,
    GameId INTEGER,
    ReviewerId INTEGER,
    FOREIGN KEY (GameId) REFERENCES VideoGames(id),
    FOREIGN KEY (ReviewerId) REFERENCES Actors(id)
);
INSERT INTO "GameReviews" VALUES(1,'Great game with a compelling story.',9,1,1);
INSERT INTO "GameReviews" VALUES(2,'Amazing graphics and immersive world.',10,2,2);
INSERT INTO "GameReviews" VALUES(3,'An epic adventure worth playing.',8,3,3);
INSERT INTO "GameReviews" VALUES(4,'A fantastic game that keeps you hooked.',9,4,4);
INSERT INTO "GameReviews" VALUES(5,'One of the best RPGs I have ever played.',10,5,5);
CREATE TABLE MovieRoles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ActorId INTEGER,
    MovieId INTEGER,
    RoleType VARCHAR(100),
    FOREIGN KEY (ActorId) REFERENCES Actors(id),
    FOREIGN KEY (MovieId) REFERENCES Movies(id)
);
INSERT INTO "MovieRoles" VALUES(1,1,1,'Leading Role');
INSERT INTO "MovieRoles" VALUES(2,2,2,'Supporting Role');
INSERT INTO "MovieRoles" VALUES(3,3,3,'Leading Role');
INSERT INTO "MovieRoles" VALUES(4,4,4,'Supporting Role');
INSERT INTO "MovieRoles" VALUES(5,5,5,'Leading Role');
CREATE TABLE Movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title VARCHAR(255),
    YearReleased INT,
    DirectorId INTEGER,
    FOREIGN KEY (DirectorId) REFERENCES Directors(id)
);
INSERT INTO "Movies" VALUES(1,'Inception',2010,4);
INSERT INTO "Movies" VALUES(2,'Interstellar',2014,5);
INSERT INTO "Movies" VALUES(3,'Titanic',1997,2);
INSERT INTO "Movies" VALUES(4,'The Matrix',1999,6);
INSERT INTO "Movies" VALUES(5,'Fight Club',1999,7);
CREATE TABLE VideoGames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Release_Year INT,
    Release_Date TEXT,
    Name VARCHAR(255),
    Platform VARCHAR(100)
);
INSERT INTO "VideoGames" VALUES(1,2018,'2018-06-12','God of War','PlayStation 4');
INSERT INTO "VideoGames" VALUES(2,2013,'2013-11-02','Grand Theft Auto V','PlayStation 3');
INSERT INTO "VideoGames" VALUES(3,2017,'2017-11-07','The Legend of Zelda: Breath of the Wild','Nintendo Switch');
INSERT INTO "VideoGames" VALUES(4,2019,'2019-04-25','Anthem','Microsoft Windows');
INSERT INTO "VideoGames" VALUES(5,2016,'2016-08-23','Overwatch','PlayStation 4');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('VideoGames',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
INSERT INTO "sqlite_sequence" VALUES('MovieRoles',5);
INSERT INTO "sqlite_sequence" VALUES('GameReviews',5);
COMMIT;
