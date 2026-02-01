BEGIN TRANSACTION;
CREATE TABLE Movie(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Director TEXT,
    Release_year INT 
);
INSERT INTO "Movie" VALUES(1,'Inception','Christopher Nolan',2010);
INSERT INTO "Movie" VALUES(2,'The Dark Knight','Christopher Nolan',2008);
CREATE TABLE Music_Album(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Artist TEXT,
    Album_name TEXT,
    Release_year INT 
);
INSERT INTO "Music_Album" VALUES(1,'Adele','21',2011);
INSERT INTO "Music_Album" VALUES(2,'Beyonce','Lemonade',2016);
CREATE TABLE Romance_Book(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Author TEXT,
    Year_of_publication INT ,
    Genre TEXT 
);
INSERT INTO "Romance_Book" VALUES(1,'Pride and Prejudice','Jane Austen',2018,'Romance');
INSERT INTO "Romance_Book" VALUES(2,'Gone With The Wind','Margaret Mitchell',1939,'Romance');
CREATE TABLE Science_Fiction_Book(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Author TEXT,
    Year_of_publication INT ,
    Genre TEXT 
);
INSERT INTO "Science_Fiction_Book" VALUES(1,'The Three Body Problem','Liu Cixin',2018,'Science Fiction');
INSERT INTO "Science_Fiction_Book" VALUES(2,'Dune','Frank Herbert',1965,'Science Fiction');
CREATE TABLE VideoGame(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Game_Name TEXT,
    Developer TEXT,
    Release_year INT 
);
INSERT INTO "VideoGame" VALUES(1,'Red Dead Redemption 2','Rockstar Games',2018);
INSERT INTO "VideoGame" VALUES(2,'Grand Theft Auto V','Rockstar North',2013);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Science_Fiction_Book',2);
INSERT INTO "sqlite_sequence" VALUES('Romance_Book',2);
INSERT INTO "sqlite_sequence" VALUES('Movie',2);
INSERT INTO "sqlite_sequence" VALUES('Music_Album',2);
INSERT INTO "sqlite_sequence" VALUES('VideoGame',2);
COMMIT;
