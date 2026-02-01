BEGIN TRANSACTION;
CREATE TABLE AlbumTracks (
    id INTEGER PRIMARY KEY,
    trackNumber INTEGER,
    title TEXT,
    duration TIME);
INSERT INTO "AlbumTracks" VALUES(1,1,'Track 1','3:00');
INSERT INTO "AlbumTracks" VALUES(2,2,'Track 2','3:30');
CREATE TABLE Albums (
    id INTEGER PRIMARY KEY,
    albumTitle TEXT,
    releaseYear INTEGER);
INSERT INTO "Albums" VALUES(1,'My First Album',2000);
INSERT INTO "Albums" VALUES(2,'The Best of Me',2010);
CREATE TABLE BluegrassSingers (
    id INTEGER PRIMARY KEY,
    singerName TEXT,
    bluegrassStyle BOOLEAN DEFAULT FALSE,
    foreignKey INTEGER REFERENCES Instruments(id));
INSERT INTO "BluegrassSingers" VALUES(1,'Tom Brown',1,1);
INSERT INTO "BluegrassSingers" VALUES(2,'Emily White',0,2);
CREATE TABLE CountrySingers (
    id INTEGER PRIMARY KEY,
    singerName TEXT,
    country TEXT,
    foreignKey INTEGER REFERENCES Instruments(id));
INSERT INTO "CountrySingers" VALUES(1,'John Doe','USA',1);
INSERT INTO "CountrySingers" VALUES(2,'Jane Smith','Canada',2);
CREATE TABLE Instruments (
    id INTEGER PRIMARY KEY,
    instrumentName TEXT UNIQUE);
INSERT INTO "Instruments" VALUES(1,'Guitar');
INSERT INTO "Instruments" VALUES(2,'Piano');
CREATE TABLE SingsOnAlbum (
    id INTEGER PRIMARY KEY,
    singerId INTEGER,
    albumId INTEGER,
    FOREIGN KEY (singerId) REFERENCES CountrySingers(id),
    FOREIGN KEY (albumId) REFERENCES Albums(id));
INSERT INTO "SingsOnAlbum" VALUES(1,1,1);
INSERT INTO "SingsOnAlbum" VALUES(2,2,1);
INSERT INTO "SingsOnAlbum" VALUES(3,3,2);
INSERT INTO "SingsOnAlbum" VALUES(4,4,2);
COMMIT;
