BEGIN TRANSACTION;
CREATE TABLE Albums (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    album_title TEXT,
    release_year INT,
    artist_id INTEGER,
    FOREIGN KEY(artist_id) REFERENCES Musicians(id));
INSERT INTO "Albums" VALUES(1,'First Album',2010,1);
INSERT INTO "Albums" VALUES(2,'Second Album',2011,1);
INSERT INTO "Albums" VALUES(3,'Third Album',2012,2);
INSERT INTO "Albums" VALUES(4,'Fourth Album',2013,2);
INSERT INTO "Albums" VALUES(5,'Fifth Album',2014,3);
CREATE TABLE Instruments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_name TEXT);
INSERT INTO "Instruments" VALUES(1,'Guitar');
INSERT INTO "Instruments" VALUES(2,'Violin');
INSERT INTO "Instruments" VALUES(3,'Piano');
INSERT INTO "Instruments" VALUES(4,'Drums');
INSERT INTO "Instruments" VALUES(5,'Bass Guitar');
CREATE TABLE MusicianInstrument (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    musician_id INTEGER,
    instrument_id INTEGER,
    FOREIGN KEY(musician_id) REFERENCES Musicians(id),
    FOREIGN KEY(instrument_id) REFERENCES Instruments(id));
INSERT INTO "MusicianInstrument" VALUES(1,1,1);
INSERT INTO "MusicianInstrument" VALUES(2,1,2);
INSERT INTO "MusicianInstrument" VALUES(3,2,3);
INSERT INTO "MusicianInstrument" VALUES(4,3,1);
INSERT INTO "MusicianInstrument" VALUES(5,4,4);
CREATE TABLE Musicians (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    musician_type TEXT);
INSERT INTO "Musicians" VALUES(1,'John Doe','Country Singer');
INSERT INTO "Musicians" VALUES(2,'Jane Smith','Country Singer');
INSERT INTO "Musicians" VALUES(3,'Bob Johnson','Bluegrass Singer');
INSERT INTO "Musicians" VALUES(4,'Alice Brown','Bluegrass Singer');
INSERT INTO "Musicians" VALUES(5,'Tom Green','Rock Star');
INSERT INTO "Musicians" VALUES(6,'Emily White','Pop Singer');
CREATE TABLE SongAlbum (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER,
    album_id INTEGER,
    FOREIGN KEY(album_id) REFERENCES Albums(id),
    FOREIGN KEY(song_id) REFERENCES Songs(id));
INSERT INTO "SongAlbum" VALUES(1,1,1);
INSERT INTO "SongAlbum" VALUES(2,2,2);
INSERT INTO "SongAlbum" VALUES(3,3,3);
INSERT INTO "SongAlbum" VALUES(4,4,4);
INSERT INTO "SongAlbum" VALUES(5,5,5);
CREATE TABLE Songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_title TEXT,
    release_date DATE);
INSERT INTO "Songs" VALUES(1,'My Favorite Song','2018-07-01');
INSERT INTO "Songs" VALUES(2,'Another Great Song','2019-06-30');
INSERT INTO "Songs" VALUES(3,'Awesome Tune','2020-05-15');
INSERT INTO "Songs" VALUES(4,'Amazing Melody','2021-04-10');
INSERT INTO "Songs" VALUES(5,'Fantastic Track','2022-03-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Musicians',6);
INSERT INTO "sqlite_sequence" VALUES('Instruments',5);
INSERT INTO "sqlite_sequence" VALUES('Songs',5);
INSERT INTO "sqlite_sequence" VALUES('Albums',5);
INSERT INTO "sqlite_sequence" VALUES('SongAlbum',5);
INSERT INTO "sqlite_sequence" VALUES('MusicianInstrument',5);
COMMIT;
