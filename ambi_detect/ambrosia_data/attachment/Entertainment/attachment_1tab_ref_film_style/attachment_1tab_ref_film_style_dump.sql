BEGIN TRANSACTION;
CREATE TABLE CastMembers(
    MemberId INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieId INTEGER,
    ActorName TEXT,
    CharacterPlayed TEXT,
    FOREIGN KEY (MovieId) REFERENCES Movies(MovieId));
INSERT INTO "CastMembers" VALUES(1,1,'Alice Johnson','Lead Protagonist');
INSERT INTO "CastMembers" VALUES(2,1,'Bob Brown','Supporting Role');
INSERT INTO "CastMembers" VALUES(3,2,'Charlie Davis','Love Interest');
INSERT INTO "CastMembers" VALUES(4,3,'David Wilson','Protagonist');
INSERT INTO "CastMembers" VALUES(5,4,'Emma Thompson','Antagonist');
CREATE TABLE Directors(
    DirectorId INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT);
INSERT INTO "Directors" VALUES(1,'Sofia Martinez');
INSERT INTO "Directors" VALUES(2,'John Doe');
INSERT INTO "Directors" VALUES(3,'Jane Smith');
CREATE TABLE MovieDetails(
    DetailID INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieId INTEGER,
    ReleaseYear INT,
    Budget NUMERIC,
    Revenue NUMERIC,
    Duration VARCHAR(50),
    Rating FLOAT,
    FOREIGN KEY (MovieId) REFERENCES Movies(MovieId));
INSERT INTO "MovieDetails" VALUES(1,1,2021,10000000,15000000,'120 minutes',8.5);
INSERT INTO "MovieDetails" VALUES(2,2,2020,7500000,9000000,'95 minutes',7.8);
INSERT INTO "MovieDetails" VALUES(3,3,2021,15000000,25000000,'135 minutes',9.2);
INSERT INTO "MovieDetails" VALUES(4,4,2022,12000000,16000000,'100 minutes',8.0);
INSERT INTO "MovieDetails" VALUES(5,5,2020,10000000,12000000,'115 minutes',7.5);
CREATE TABLE Movies(
    MovieId INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    FilmStyle TEXT,
    DirectorId INTEGER,
    FOREIGN KEY (DirectorId) REFERENCES Directors(DirectorId));
INSERT INTO "Movies" VALUES(1,'The Lost City','Adventure Movie',1);
INSERT INTO "Movies" VALUES(2,'Racing Hearts','Romantic Comedy',2);
INSERT INTO "Movies" VALUES(3,'Guns of Glory','Action Movie',1);
INSERT INTO "Movies" VALUES(4,'Alien Invasion','Science Fiction',3);
INSERT INTO "Movies" VALUES(5,'Epic Adventure','Adventure Movie',2);
CREATE TABLE Reviews(
    ReviewId INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieId INTEGER,
    UserId INTEGER,
    Comment TEXT,
    Stars RATING ,
    FOREIGN KEY (MovieId) REFERENCES Movies(MovieId),
    FOREIGN KEY (UserId) REFERENCES Users(UserId));
INSERT INTO "Reviews" VALUES(1,1,1,'Great adventure! Loved it.',5);
INSERT INTO "Reviews" VALUES(2,2,1,'Not my cup of tea...',3);
INSERT INTO "Reviews" VALUES(3,3,2,'Exciting action sequences!',4);
INSERT INTO "Reviews" VALUES(4,4,3,'Predictable plot...',2);
INSERT INTO "Reviews" VALUES(5,5,4,'Fun watch overall!',3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',3);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
INSERT INTO "sqlite_sequence" VALUES('MovieDetails',5);
INSERT INTO "sqlite_sequence" VALUES('CastMembers',5);
INSERT INTO "sqlite_sequence" VALUES('Reviews',5);
COMMIT;
