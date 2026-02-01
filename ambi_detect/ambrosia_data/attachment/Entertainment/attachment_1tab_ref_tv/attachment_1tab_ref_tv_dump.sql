BEGIN TRANSACTION;
CREATE TABLE EpisodeCounts(
    Show_ID INTEGER,
    Number_of_Episodes INTEGER,
    FOREIGN KEY (Show_ID) REFERENCES TVShows(Id)
);
INSERT INTO "EpisodeCounts" VALUES(1,10);
INSERT INTO "EpisodeCounts" VALUES(2,12);
INSERT INTO "EpisodeCounts" VALUES(3,10);
INSERT INTO "EpisodeCounts" VALUES(4,8);
CREATE TABLE Genres(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Genres" VALUES(1,'Documentary');
INSERT INTO "Genres" VALUES(2,'Drama');
INSERT INTO "Genres" VALUES(3,'Comedy');
INSERT INTO "Genres" VALUES(4,'Action');
INSERT INTO "Genres" VALUES(5,'Fantasy');
CREATE TABLE Ratings(
    Show_ID INTEGER,
    RatingValue FLOAT,
    ReviewText TEXT,
    RaterName TEXT,
    RatedOn TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Show_ID) REFERENCES TVShows(Id)
);
INSERT INTO "Ratings" VALUES(1,9.0,'Great documentary!','JohnDoe','2024-03-13 10:56:18');
INSERT INTO "Ratings" VALUES(2,8.5,'Interesting series but could have been better.','JaneSmith','2024-03-13 10:56:18');
INSERT INTO "Ratings" VALUES(3,8.2,'Amazing mini-series!','AliceBrown','2024-03-13 10:56:18');
INSERT INTO "Ratings" VALUES(4,9.1,'Loved it! Can''t wait for the next season.','RobertWhite','2024-03-13 10:56:18');
CREATE TABLE TVShows(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Type TEXT ,
    ReleaseDate DATE,
    GENRE_ID INTEGER,
    FOREIGN KEY (GENRE_ID) REFERENCES Genres(Id)
);
INSERT INTO "TVShows" VALUES(1,'The Planets','Documentary Series','2023-01-01',1);
INSERT INTO "TVShows" VALUES(2,'Nature Calls','Documentary Series','2023-07-01',1);
INSERT INTO "TVShows" VALUES(3,'A Mini Adventure','Mini-Series','2023-06-01',2);
INSERT INTO "TVShows" VALUES(4,'Time Travelers','Mini-Series','2023-08-01',2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Genres',5);
INSERT INTO "sqlite_sequence" VALUES('TVShows',4);
COMMIT;
