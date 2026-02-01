BEGIN TRANSACTION;
CREATE TABLE Albums (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseDate DATE,
    artistId INTEGER,
    FOREIGN KEY(artistId) REFERENCES Artists(id)
);
INSERT INTO "Albums" VALUES(1,'Album1','2001-01-01',1);
INSERT INTO "Albums" VALUES(2,'Album2','2002-06-15',2);
INSERT INTO "Albums" VALUES(3,'Album3','2003-07-20',3);
INSERT INTO "Albums" VALUES(4,'Album4','2004-08-10',4);
INSERT INTO "Albums" VALUES(5,'Album5','2005-09-25',5);
CREATE TABLE Artists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    genre TEXT,
    birthdate DATE,
    country TEXT
);
INSERT INTO "Artists" VALUES(1,'Artist1','Pop','2000-01-01','USA');
INSERT INTO "Artists" VALUES(2,'Artist2','Rock','1985-06-15','Canada');
INSERT INTO "Artists" VALUES(3,'Artist3','Hip Hop','1990-07-20','UK');
INSERT INTO "Artists" VALUES(4,'Artist4','Jazz','1970-08-10','France');
INSERT INTO "Artists" VALUES(5,'Artist5','Classical','1965-09-25','Germany');
CREATE TABLE "DanceTable" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Origin TEXT,
    DanceName TEXT,
    Popularity INT,
    Tradition TEXT
);
INSERT INTO "DanceTable" VALUES(1,'Ghanaian','Kpanlogo',90,'Spiritual and social event dance');
INSERT INTO "DanceTable" VALUES(2,'Korean','Fan Dance',80,'Cultural performance');
INSERT INTO "DanceTable" VALUES(3,'Spanish','Flamenco',70,'Expressive art form');
INSERT INTO "DanceTable" VALUES(4,'Argentine','Tango',85,'Ballroom and social dance');
INSERT INTO "DanceTable" VALUES(5,'American','Breakdancing',75,'Street dance');
CREATE TABLE Songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    length TIME,
    albumId INTEGER,
    FOREIGN KEY(albumId) REFERENCES Albums(id)
);
INSERT INTO "Songs" VALUES(1,'Song1','03:00:00',1);
INSERT INTO "Songs" VALUES(2,'Song2','04:00:00',2);
INSERT INTO "Songs" VALUES(3,'Song3','05:00:00',3);
INSERT INTO "Songs" VALUES(4,'Song4','06:00:00',4);
INSERT INTO "Songs" VALUES(5,'Song5','07:00:00',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Artists',5);
INSERT INTO "sqlite_sequence" VALUES('Albums',5);
INSERT INTO "sqlite_sequence" VALUES('Songs',5);
INSERT INTO "sqlite_sequence" VALUES('DanceTable',5);
COMMIT;
