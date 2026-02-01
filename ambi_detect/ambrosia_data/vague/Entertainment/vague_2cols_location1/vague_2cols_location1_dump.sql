BEGIN TRANSACTION;
CREATE TABLE Albums (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseDate DATE,
    ArtistID INTEGER,
    FOREIGN KEY(ArtistID) REFERENCES Artists(id));
INSERT INTO "Albums" VALUES(1,'Lemonade','2016-04-23',1);
CREATE TABLE Artists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Nationality TEXT,
    Occupation TEXT);
INSERT INTO "Artists" VALUES(1,'Beyonce','USA','Musician');
CREATE TABLE Festivals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Country TEXT,
    City TEXT,
    FestivalName TEXT,
    Date DATE,
    Genre TEXT);
INSERT INTO "Festivals" VALUES(1,'USA','Los Angeles','Coachella Music and Arts Festival','2023-04-13','Music');
CREATE TABLE MovieTitles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INTEGER,
    Director TEXT,
    LeadActor TEXT);
INSERT INTO "MovieTitles" VALUES(1,'Inception',2010,'Christopher Nolan','Leonardo DiCaprio');
CREATE TABLE MoviesAndSongs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieTitleID INTEGER,
    SongID INTEGER,
    FOREIGN KEY(MovieTitleID) REFERENCES MovieTitles(id),
    FOREIGN KEY(SongID) REFERENCES Songs(id));
INSERT INTO "MoviesAndSongs" VALUES(1,1,1);
CREATE TABLE Songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Length NUMERIC,
    AlbumID INTEGER,
    FOREIGN KEY(AlbumID) REFERENCES Albums(id));
INSERT INTO "Songs" VALUES(1,'Formation','03:38',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Festivals',1);
INSERT INTO "sqlite_sequence" VALUES('Artists',1);
INSERT INTO "sqlite_sequence" VALUES('Albums',1);
INSERT INTO "sqlite_sequence" VALUES('Songs',1);
INSERT INTO "sqlite_sequence" VALUES('MovieTitles',1);
INSERT INTO "sqlite_sequence" VALUES('MoviesAndSongs',1);
COMMIT;
