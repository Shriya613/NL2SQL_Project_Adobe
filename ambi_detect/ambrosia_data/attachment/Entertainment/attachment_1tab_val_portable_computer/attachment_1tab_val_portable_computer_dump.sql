BEGIN TRANSACTION;
CREATE TABLE Devices(
    DeviceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Portable_Computer TEXT,
    Operating_System TEXT,
    ReleaseDate DATE,
    Price REAL 
);
INSERT INTO "Devices" VALUES(1,'Tablet','Windows 10','2020-07-30',999.99);
INSERT INTO "Devices" VALUES(2,'Tablet','iOS','2021-03-26',899.99);
INSERT INTO "Devices" VALUES(3,'Laptop','Windows 10','2020-05-01',1199.99);
INSERT INTO "Devices" VALUES(4,'Laptop','macOS','2020-11-10',1399.99);
CREATE TABLE Games(
    GameID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title VARCHAR(255),
    Developer VARCHAR(255),
    Genre VARCHAR(255),
    PlatformId INTEGER,
    FOREIGN KEY (PlatformId) REFERENCES Devices(DeviceID)
);
INSERT INTO "Games" VALUES(1,'Game A','Developer XYZ','Action',1);
INSERT INTO "Games" VALUES(2,'Game B','Developer ABC','Strategy',2);
CREATE TABLE Movies(
    MovieID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title VARCHAR(255),
    Director VARCHAR(255),
    ReleaseYear INTEGER ,
    Runtime INTEGER ,
    Rating REAL 
);
INSERT INTO "Movies" VALUES(1,'Movie A','Director PQR',2019,120,7.5);
INSERT INTO "Movies" VALUES(2,'Movie B','Director STU',2020,135,8.0);
CREATE TABLE MusicAlbums(
    AlbumID INTEGER PRIMARY KEY AUTOINCREMENT,
    ArtistName VARCHAR(255),
    AlbumTitle VARCHAR(255),
    YearReleased INTEGER ,
    NumTracks INTEGER 
);
INSERT INTO "MusicAlbums" VALUES(1,'Artist WXY','Album ZUV',2021,12);
INSERT INTO "MusicAlbums" VALUES(2,'Artist ABC','Album DEF',2020,10);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Devices',4);
INSERT INTO "sqlite_sequence" VALUES('Games',2);
INSERT INTO "sqlite_sequence" VALUES('Movies',2);
INSERT INTO "sqlite_sequence" VALUES('MusicAlbums',2);
COMMIT;
