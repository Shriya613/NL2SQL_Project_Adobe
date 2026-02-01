BEGIN TRANSACTION;
CREATE TABLE Genre (
    id INTEGER PRIMARY KEY,
    Name TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Action');
INSERT INTO "Genre" VALUES(2,'Comedy');
INSERT INTO "Genre" VALUES(3,'Horror');
INSERT INTO "Genre" VALUES(4,'Documentary');
CREATE TABLE Platforms (
    id INTEGER PRIMARY KEY,
    Name TEXT UNIQUE
);
INSERT INTO "Platforms" VALUES(1,'Netflix');
INSERT INTO "Platforms" VALUES(2,'Hulu');
INSERT INTO "Platforms" VALUES(3,'Disney+');
INSERT INTO "Platforms" VALUES(4,'Amazon Prime Video');
CREATE TABLE Ratings (
    id INTEGER PRIMARY KEY,
    RateValue FLOAT 
);
INSERT INTO "Ratings" VALUES(1,9.0);
INSERT INTO "Ratings" VALUES(2,8.9);
INSERT INTO "Ratings" VALUES(3,8.6);
INSERT INTO "Ratings" VALUES(4,8.4);
INSERT INTO "Ratings" VALUES(5,8.1);
CREATE TABLE "TV_Shows" (
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Number_Of_Seasons INT,
    Series TEXT,
    GenreID INTEGER,
    PlatformID INTEGER,
    RatingID INTEGER,
    FOREIGN KEY(GenreID) REFERENCES Genre(id),
    FOREIGN KEY(PlatformID) REFERENCES Platforms(id),
    FOREIGN KEY(RatingID) REFERENCES Ratings(id)
);
INSERT INTO "TV_Shows" VALUES(1,'Breaking Bad',6,'Drama',2,3,4);
INSERT INTO "TV_Shows" VALUES(2,'Game of Thrones',7,'Fantasy Drama',3,3,5);
INSERT INTO "TV_Shows" VALUES(3,'Stranger Things',3,'Mystery Sci-Fi',1,4,3);
INSERT INTO "TV_Shows" VALUES(4,'The Witcher',1,'Adventure Fantasy',4,4,2);
INSERT INTO "TV_Shows" VALUES(5,'Ozark',3,'Crime Drama',2,3,4);
CREATE TABLE User_Reviews (
    id INTEGER PRIMARY KEY,
    UserID INTEGER,
    ShowID INTEGER,
    ReviewText TEXT,
    RatingID INTEGER,
    FOREIGN KEY(UserID) REFERENCES Users(id),
    FOREIGN KEY(ShowID) REFERENCES TV_Shows(id),
    FOREIGN KEY(RatingID) REFERENCES Ratings(id)
);
INSERT INTO "User_Reviews" VALUES(1,1,1,'Great show!',1);
INSERT INTO "User_Reviews" VALUES(2,2,2,'Best series ever.',2);
INSERT INTO "User_Reviews" VALUES(3,3,3,'Loved it! Can''t wait for season 4.',3);
INSERT INTO "User_Reviews" VALUES(4,4,4,'Not a fan of this one.',3);
INSERT INTO "User_Reviews" VALUES(5,5,5,'Interesting storyline and characters.',4);
CREATE TABLE Users (
    id INTEGER PRIMARY KEY,
    Username TEXT,
    Email TEXT,
    Password TEXT
);
INSERT INTO "Users" VALUES(1,'user1','user1@example.com','password1');
INSERT INTO "Users" VALUES(2,'user2','user2@example.com','password2');
INSERT INTO "Users" VALUES(3,'user3','user3@example.com','password3');
INSERT INTO "Users" VALUES(4,'user4','user4@example.com','password4');
INSERT INTO "Users" VALUES(5,'user5','user5@example.com','password5');
COMMIT;
