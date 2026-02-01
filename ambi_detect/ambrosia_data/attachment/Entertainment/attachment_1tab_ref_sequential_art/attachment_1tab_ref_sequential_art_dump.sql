BEGIN TRANSACTION;
CREATE TABLE ArtWork(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    publishDate DATE ,
    publisherId INTEGER,
    sequentialArtTypeId INTEGER,
    FOREIGN KEY(publisherId) REFERENCES Publisher(id),
    FOREIGN KEY(sequentialArtTypeId) REFERENCES SequentialArtType(id)
);
INSERT INTO "ArtWork" VALUES(1,'X-Men #1','2023-01-01',1,1);
INSERT INTO "ArtWork" VALUES(2,'Batman #1','2023-01-15',2,1);
INSERT INTO "ArtWork" VALUES(3,'Spider-Man #1','2023-02-01',3,1);
INSERT INTO "ArtWork" VALUES(4,'The Walking Dead #1','2023-02-15',4,1);
INSERT INTO "ArtWork" VALUES(5,'Saga #1','2023-03-01',1,2);
CREATE TABLE Artist(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    artWorkId INTEGER,
    FOREIGN KEY(artWorkId) REFERENCES ArtWork(id)
);
INSERT INTO "Artist" VALUES(1,'John','Doe',1);
INSERT INTO "Artist" VALUES(2,'Jane','Smith',2);
INSERT INTO "Artist" VALUES(3,'Alice','Johnson',3);
INSERT INTO "Artist" VALUES(4,'Bob','Williams',4);
INSERT INTO "Artist" VALUES(5,'Charlie','Brown',5);
CREATE TABLE Publisher(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Publisher" VALUES(1,'BOOM Studios');
INSERT INTO "Publisher" VALUES(2,'Marvel Comics');
INSERT INTO "Publisher" VALUES(3,'DC Comics');
INSERT INTO "Publisher" VALUES(4,'IDW Publishing');
INSERT INTO "Publisher" VALUES(5,'Image Comics');
CREATE TABLE SequentialArtType(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT UNIQUE
);
INSERT INTO "SequentialArtType" VALUES(1,'Comic Book');
INSERT INTO "SequentialArtType" VALUES(2,'Graphic Novel');
INSERT INTO "SequentialArtType" VALUES(3,'Manga');
INSERT INTO "SequentialArtType" VALUES(4,'Webcomic');
INSERT INTO "SequentialArtType" VALUES(5,'Digital Comic');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Publisher',5);
INSERT INTO "sqlite_sequence" VALUES('SequentialArtType',5);
INSERT INTO "sqlite_sequence" VALUES('ArtWork',5);
INSERT INTO "sqlite_sequence" VALUES('Artist',5);
COMMIT;
