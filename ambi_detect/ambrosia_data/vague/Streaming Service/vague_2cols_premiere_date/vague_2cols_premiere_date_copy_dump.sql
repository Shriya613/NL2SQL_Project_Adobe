BEGIN TRANSACTION;
CREATE TABLE "Content" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT ,
    title TEXT,
    genre TEXT,
    creator TEXT,
    numEpisodesOrDuration INTEGER,
    Premiere_Date INTEGER,
    season INT DEFAULT 1,
    FOREIGN KEY(creator) REFERENCES Creators(id));
INSERT INTO "Content" VALUES(1,'TV Series','Stranger Things','Science Fiction','The Duffer Brothers',8,7,1);
INSERT INTO "Content" VALUES(2,'Film','Inception','Action, Adventure, Sci-Fi','Christopher Nolan',128,7,NULL);
CREATE TABLE Creators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthYear INTEGER);
INSERT INTO "Creators" VALUES(1,'Quentin Tarantino',1963);
INSERT INTO "Creators" VALUES(2,'James Cameron',1954);
CREATE TABLE Movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    genre TEXT,
    director TEXT,
    runtime INTEGER,
    release_date DATE);
INSERT INTO "Movies" VALUES(1,'The Shawshank Redemption','Drama','Frank Darabont',142,'1994-09-23');
INSERT INTO "Movies" VALUES(2,'The Godfather','Crime','Francis Ford Coppola',175,'1972-03-24');
CREATE TABLE Reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userId INTEGER,
    contentId INTEGER,
    rating INTEGER ,
    reviewText TEXT,
    FOREIGN KEY(userId) REFERENCES Users(id),
    FOREIGN KEY(contentId) REFERENCES Content(id));
INSERT INTO "Reviews" VALUES(1,1,1,5,'Great movie with a powerful story.');
INSERT INTO "Reviews" VALUES(2,2,2,4,'Awesome series with amazing characters.');
CREATE TABLE Series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    genre TEXT,
    creator TEXT,
    episodeCount INTEGER,
    premiereDate DATE);
INSERT INTO "Series" VALUES(1,'Breaking Bad','Crime Drama','Vince Gilligan',62,'2008-01-20');
INSERT INTO "Series" VALUES(2,'Game of Thrones','Fantasy','David Benioff and D.B. Weiss',60,'2011-04-17');
CREATE TABLE Users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT,
    email TEXT,
    subscriptionType TEXT );
INSERT INTO "Users" VALUES(1,'user1','password1','user1@example.com','Premium');
INSERT INTO "Users" VALUES(2,'user2','password2','user2@example.com','Standard');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Movies',2);
INSERT INTO "sqlite_sequence" VALUES('Series',2);
INSERT INTO "sqlite_sequence" VALUES('Users',2);
INSERT INTO "sqlite_sequence" VALUES('Reviews',2);
INSERT INTO "sqlite_sequence" VALUES('Creators',2);
INSERT INTO "sqlite_sequence" VALUES('Content',2);
COMMIT;
