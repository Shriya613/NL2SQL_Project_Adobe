BEGIN TRANSACTION;
CREATE TABLE Albums(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_date DATE,
    artist_id INTEGER,
    FOREIGN KEY (artist_id) REFERENCES Artists(id)
);
INSERT INTO "Albums" VALUES(1,'2021-01-01',1);
INSERT INTO "Albums" VALUES(2,'2021-02-01',2);
INSERT INTO "Albums" VALUES(3,'2021-03-01',3);
INSERT INTO "Albums" VALUES(4,'2021-04-01',4);
INSERT INTO "Albums" VALUES(5,'2021-05-01',5);
CREATE TABLE Artists(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Artists" VALUES(1,'Artist1');
INSERT INTO "Artists" VALUES(2,'Artist2');
INSERT INTO "Artists" VALUES(3,'Artist3');
INSERT INTO "Artists" VALUES(4,'Artist4');
INSERT INTO "Artists" VALUES(5,'Artist5');
CREATE TABLE Concerts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue TEXT,
    date TIMESTAMP
);
INSERT INTO "Concerts" VALUES(1,'Madison Square Garden','2021-06-01 18:00:00');
INSERT INTO "Concerts" VALUES(2,'Carnegie Hall','2021-07-01 19:00:00');
INSERT INTO "Concerts" VALUES(3,'The Hollywood Bowl','2021-08-01 20:00:00');
INSERT INTO "Concerts" VALUES(4,'Wembley Stadium','2021-09-01 21:00:00');
INSERT INTO "Concerts" VALUES(5,'Stadium Australia','2021-10-01 22:00:00');
CREATE TABLE Music_Genres(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE 
);
INSERT INTO "Music_Genres" VALUES(1,'Pop Song');
INSERT INTO "Music_Genres" VALUES(2,'Hip Hop Song');
CREATE TABLE Songs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    artist_id INTEGER,
    album_id INTEGER,
    concert_id INTEGER,
    music_genre_id INTEGER,
    FOREIGN KEY (artist_id) REFERENCES Performers(id),
    FOREIGN KEY (album_id) REFERENCES Albums(id),
    FOREIGN KEY (concert_id) REFERENCES Concerts(id),
    FOREIGN KEY (music_genre_id) REFERENCES Music_Genres(id)
);
INSERT INTO "Songs" VALUES(1,'Tune1',1,1,1,1);
INSERT INTO "Songs" VALUES(2,'Tune2',2,2,3,1);
INSERT INTO "Songs" VALUES(3,'Rap1',3,3,1,2);
INSERT INTO "Songs" VALUES(4,'Rap2',4,4,5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Music_Genres',2);
INSERT INTO "sqlite_sequence" VALUES('Songs',4);
INSERT INTO "sqlite_sequence" VALUES('Artists',5);
INSERT INTO "sqlite_sequence" VALUES('Albums',5);
INSERT INTO "sqlite_sequence" VALUES('Concerts',5);
COMMIT;
