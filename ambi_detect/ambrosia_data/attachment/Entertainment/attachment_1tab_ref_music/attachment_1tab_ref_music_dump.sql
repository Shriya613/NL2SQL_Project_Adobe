BEGIN TRANSACTION;
CREATE TABLE "Album" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE
);
INSERT INTO "Album" VALUES(1,NULL,'2000-01-01');
INSERT INTO "Album" VALUES(2,NULL,'2001-06-01');
INSERT INTO "Album" VALUES(3,NULL,'2002-11-01');
INSERT INTO "Album" VALUES(4,NULL,'2003-05-01');
INSERT INTO "Album" VALUES(5,NULL,'2004-10-01');
CREATE TABLE Instrument (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT UNIQUE
);
INSERT INTO "Instrument" VALUES(1,'Guitar');
INSERT INTO "Instrument" VALUES(2,'Piano');
INSERT INTO "Instrument" VALUES(3,'Drums');
INSERT INTO "Instrument" VALUES(4,'Violin');
CREATE TABLE "Music" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    length TIME,
    MUSIC_TYPE TEXT ,
    INSTRUMENT_id INTEGER,
    ALBUM_id INTEGER,
    SINGER_id INTEGER,
    FOREIGN KEY (ALBUM_id) REFERENCES Album (id),
    FOREIGN KEY (SINGER_id) REFERENCES Singer (id),
    FOREIGN KEY (INSTRUMENT_id) REFERENCES Instrument (id)
);
INSERT INTO "Music" VALUES(1,'Blues Song','03:00','Single',1,1,1);
INSERT INTO "Music" VALUES(2,'Blues Song','03:00','Single',2,2,2);
INSERT INTO "Music" VALUES(3,'Folk Song','03:00','Single',3,3,1);
INSERT INTO "Music" VALUES(4,'Folk Song','03:00','Single',4,4,3);
INSERT INTO "Music" VALUES(5,'Country','03:00','Single',5,5,4);
CREATE TABLE Singer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE,
    death_date DATE
);
INSERT INTO "Singer" VALUES(1,'John','Hurt','1923-06-08','2007-01-27');
INSERT INTO "Singer" VALUES(2,'Bob','Smith','1960-07-01',NULL);
INSERT INTO "Singer" VALUES(3,'Alice','Johnson','1975-01-10',NULL);
INSERT INTO "Singer" VALUES(4,'Tom','White','1980-02-05',NULL);
INSERT INTO "Singer" VALUES(5,'Emily','Green','1992-03-15',NULL);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Instrument',4);
INSERT INTO "sqlite_sequence" VALUES('Singer',5);
INSERT INTO "sqlite_sequence" VALUES('Music',5);
INSERT INTO "sqlite_sequence" VALUES('Album',5);
COMMIT;
