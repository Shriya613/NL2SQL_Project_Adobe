BEGIN TRANSACTION;
CREATE TABLE Albums (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    artist_name TEXT
);
INSERT INTO "Albums" VALUES(1,'Thriller','1982-06-30','Michael Jackson');
INSERT INTO "Albums" VALUES(2,'Abbey Road','1969-09-26','The Beatles');
CREATE TABLE Artists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    genre TEXT,
    country TEXT
);
INSERT INTO "Artists" VALUES(1,'Michael Jackson','Pop','USA');
INSERT INTO "Artists" VALUES(2,'The Beatles','Rock','UK');
CREATE TABLE "Ballets" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    Creator TEXT
);
INSERT INTO "Ballets" VALUES(1,'Swan Lake','Pyotr Ilyich Tchaikovsky');
INSERT INTO "Ballets" VALUES(2,'Giselle','Adolphe Adam');
CREATE TABLE Concerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATETIME,
    location TEXT,
    headliner_artist_id INTEGER,
    opening_act_artist_id INTEGER,
    FOREIGN KEY(headliner_artist_id) REFERENCES Artists(id),
    FOREIGN KEY(opening_act_artist_id) REFERENCES Artists(id)
);
INSERT INTO "Concerts" VALUES(1,'2022-06-01','New York City',1,2);
INSERT INTO "Concerts" VALUES(2,'2022-06-02','Los Angeles',1,NULL);
CREATE TABLE Performances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATETIME,
    ballet_id INTEGER,
    performers_count INTEGER,
    FOREIGN KEY(ballet_id) REFERENCES Ballets(id)
);
INSERT INTO "Performances" VALUES(1,'2022-07-01',1,50);
INSERT INTO "Performances" VALUES(2,'2022-07-02',2,45);
CREATE TABLE Songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    length REAL,
    album_id INTEGER,
    FOREIGN KEY(album_id) REFERENCES Albums(id)
);
INSERT INTO "Songs" VALUES(1,'Billie Jean',4.57,1);
INSERT INTO "Songs" VALUES(2,'Hey Jude',7.1,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Albums',2);
INSERT INTO "sqlite_sequence" VALUES('Songs',2);
INSERT INTO "sqlite_sequence" VALUES('Artists',2);
INSERT INTO "sqlite_sequence" VALUES('Concerts',2);
INSERT INTO "sqlite_sequence" VALUES('Performances',2);
INSERT INTO "sqlite_sequence" VALUES('Ballets',2);
COMMIT;
