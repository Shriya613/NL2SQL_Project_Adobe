BEGIN TRANSACTION;
CREATE TABLE Albums(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    artist_id INTEGER,
    FOREIGN KEY(artist_id) REFERENCES Artists(id)
);
INSERT INTO "Albums" VALUES(1,'Jazz Collection','2022-01-01',1);
INSERT INTO "Albums" VALUES(2,'Classical Jazz','2022-07-01',2);
INSERT INTO "Albums" VALUES(3,'Guitar Masterpieces','2022-06-01',3);
INSERT INTO "Albums" VALUES(4,'Acoustic Rock','2022-08-01',4);
CREATE TABLE Artist_genre(
    artist_id INTEGER,
    genre_id INTEGER,
    years_in_industry INTEGER,
    PRIMARY KEY(artist_id, genre_id),
    FOREIGN KEY(artist_id) REFERENCES Artists(id),
    FOREIGN KEY(genre_id) REFERENCES Genres(genre_id)
);
INSERT INTO "Artist_genre" VALUES(1,1,10);
INSERT INTO "Artist_genre" VALUES(2,1,20);
INSERT INTO "Artist_genre" VALUES(3,2,10);
INSERT INTO "Artist_genre" VALUES(4,2,30);
CREATE TABLE "Artists"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Artists" VALUES(1,'John Doe');
INSERT INTO "Artists" VALUES(2,'Jane Smith');
INSERT INTO "Artists" VALUES(3,'Tom Brown');
INSERT INTO "Artists" VALUES(4,'Emily Davis');
CREATE TABLE Genres(
    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Genres" VALUES(1,'Jazz');
INSERT INTO "Genres" VALUES(2,'Rock');
CREATE TABLE Songs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    length REAL,
    album_id INTEGER,
    FOREIGN KEY(album_id) REFERENCES Albums(id)
);
INSERT INTO "Songs" VALUES(1,'Cool Blues',3.5,1);
INSERT INTO "Songs" VALUES(2,'Elegant Waltz',4.2,2);
INSERT INTO "Songs" VALUES(3,'Fast Riff',3.0,3);
INSERT INTO "Songs" VALUES(4,'Slow Ballad',4.7,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Genres',2);
INSERT INTO "sqlite_sequence" VALUES('Albums',4);
INSERT INTO "sqlite_sequence" VALUES('Songs',4);
INSERT INTO "sqlite_sequence" VALUES('Artists',4);
COMMIT;
