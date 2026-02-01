BEGIN TRANSACTION;
CREATE TABLE Albums(
    id INTEGER PRIMARY KEY,
    AlbumTitle TEXT,
    ReleaseYear INTEGER,
    Music_Genre_id INTEGER,
    FOREIGN KEY(Music_Genre_id) REFERENCES Music_Genre(id)
);
INSERT INTO "Albums" VALUES(1,'The Blues Collection',2001,1);
INSERT INTO "Albums" VALUES(2,'Layla and Other Assorted Love Songs',1970,1);
INSERT INTO "Albums" VALUES(3,'Blood on the Tracks',1975,3);
INSERT INTO "Albums" VALUES(4,'21',2011,4);
INSERT INTO "Albums" VALUES(5,'From Elvis Presley Boulevard, Memphis, Tennessee',1956,5);
CREATE TABLE Music_Genre(
    id INTEGER PRIMARY KEY,
    Genre TEXT
);
INSERT INTO "Music_Genre" VALUES(1,'Blues');
INSERT INTO "Music_Genre" VALUES(2,'Rock');
INSERT INTO "Music_Genre" VALUES(3,'Pop');
INSERT INTO "Music_Genre" VALUES(4,'Jazz');
INSERT INTO "Music_Genre" VALUES(5,'Country');
CREATE TABLE Performances(
    id INTEGER PRIMARY KEY,
    PerformanceDate DATETIME,
    Venue TEXT,
    Song_id INTEGER,
    FOREIGN KEY(Song_id) REFERENCES Songs(id)
);
INSERT INTO "Performances" VALUES(1,'2021-08-15 20:00:00','Madison Square Garden',1);
INSERT INTO "Performances" VALUES(2,'2021-08-16 20:00:00','Wembley Stadium',2);
INSERT INTO "Performances" VALUES(3,'2021-08-17 20:00:00','Staples Center',3);
INSERT INTO "Performances" VALUES(4,'2021-08-18 20:00:00','O2 Arena',4);
INSERT INTO "Performances" VALUES(5,'2021-08-19 20:00:00','Barclays Center',5);
CREATE TABLE Singers(
    id INTEGER PRIMARY KEY,
    SingerName TEXT
);
INSERT INTO "Singers" VALUES(1,'John Hurt');
INSERT INTO "Singers" VALUES(2,'Eric Clapton');
INSERT INTO "Singers" VALUES(3,'Bob Dylan');
INSERT INTO "Singers" VALUES(4,'Adele');
INSERT INTO "Singers" VALUES(5,'Elvis Presley');
CREATE TABLE Songs(
    id INTEGER PRIMARY KEY,
    SongTitle TEXT,
    Duration INTEGER,
    Music TEXT,
    Singer_id INTEGER,
    Album_id INTEGER,
    FOREIGN KEY(Singer_id) REFERENCES Singers(id),
    FOREIGN KEY(Album_id) REFERENCES Albums(id)
);
INSERT INTO "Songs" VALUES(1,'Blues Song',300,'Blues Music',1,1);
INSERT INTO "Songs" VALUES(2,'Another Blues Song',280,'Blues Music',2,1);
INSERT INTO "Songs" VALUES(3,'Folk Song',290,'Folk Music',1,2);
INSERT INTO "Songs" VALUES(4,'Yet Another Folk Song',270,'Folk Music',3,2);
INSERT INTO "Songs" VALUES(5,'Love Story',220,'Pop Music',4,3);
COMMIT;
