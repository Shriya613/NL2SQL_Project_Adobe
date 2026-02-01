BEGIN TRANSACTION;
CREATE TABLE "Albums" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Release_Timeframe INTEGER,
    AlbumName TEXT,
    Genre TEXT,
    ArtistId INTEGER,
    FOREIGN KEY(ArtistId) REFERENCES Artists(id)
);
INSERT INTO "Albums" VALUES(1,2019,'Echoes of the Past','Classical',1);
INSERT INTO "Albums" VALUES(2,2018,'Symphony No. 3','Orchestral',2);
INSERT INTO "Albums" VALUES(3,2020,'The Journey Home','Jazz',3);
INSERT INTO "Albums" VALUES(4,2017,'Piano Sonata No. 14','Pianosonata',4);
INSERT INTO "Albums" VALUES(5,2016,'Requiem Mass','Choral',5);
CREATE TABLE Artists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT,
    MusicGenre TEXT
);
INSERT INTO "Artists" VALUES(1,'John Smith','1980-06-01','USA','Classical');
INSERT INTO "Artists" VALUES(2,'Emily Johnson','1985-02-25','UK','Pop');
INSERT INTO "Artists" VALUES(3,'David Brown','1990-03-07','Canada','Rock');
INSERT INTO "Artists" VALUES(4,'Anna White','1988-08-10','Germany','Blues');
INSERT INTO "Artists" VALUES(5,'Mark Green','1992-11-12','Australia','Country');
CREATE TABLE Directors (
    DirectorName TEXT PRIMARY KEY,
    Birthdate DATE,
    Country TEXT,
    KnownFor TEXT
);
INSERT INTO "Directors" VALUES('Robert Zemeckis','1946-05-14','America','Back to the Future');
INSERT INTO "Directors" VALUES('James Cameron','1954-08-16','Canada','Avatar');
INSERT INTO "Directors" VALUES('Christopher Nolan','1970-07-30','England','Dark Knight Trilogy');
INSERT INTO "Directors" VALUES('Gabriele Muccino','1967-04-07','Italy','Pursuit of Happiness');
INSERT INTO "Directors" VALUES('Frank Darabont','1955-01-28','America','Shawshank Redemption');
CREATE TABLE FilmRatings (
    RatingID INTEGER PRIMARY KEY AUTOINCREMENT,
    MovieID INTEGER,
    Rating TEXT,
    Review TEXT,
    FOREIGN KEY(MovieID) REFERENCES Movies(MovieID)
);
INSERT INTO "FilmRatings" VALUES(1,1,'A+','Great movie! Loved it.');
INSERT INTO "FilmRatings" VALUES(2,2,'A','Beautiful film with amazing acting.');
INSERT INTO "FilmRatings" VALUES(3,3,'A-','Excellent storyline and special effects.');
INSERT INTO "FilmRatings" VALUES(4,4,'B+','Good plot but could have been better.');
INSERT INTO "FilmRatings" VALUES(5,5,'B','Interesting concept but fell short on execution.');
CREATE TABLE MovieActors (
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Nationality TEXT
);
INSERT INTO "MovieActors" VALUES(1,'Tom Hanks','Hollywood Legend','1956-07-09','American');
INSERT INTO "MovieActors" VALUES(2,'Kate Winslet','Acclaimed Actress','1975-10-05','English');
INSERT INTO "MovieActors" VALUES(3,'Leonardo DiCaprio','Versatile Star','1974-11-11','American');
INSERT INTO "MovieActors" VALUES(4,'Cate Blanchett','Talented Performer','1969-05-14','Australian');
INSERT INTO "MovieActors" VALUES(5,'Christopher Walken','Distinctive Actor','1943-03-31','American');
CREATE TABLE Movies (
    MovieID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Director TEXT,
    ReleaseDate DATE,
    Genre TEXT,
    Runtime INTEGER,
    FOREIGN KEY(Director) REFERENCES Directors(DirectorName)
);
INSERT INTO "Movies" VALUES(1,'Forest Gump','Robert Zemeckis','1994-07-06','Drama/Romance',150);
INSERT INTO "Movies" VALUES(2,'Titanic','James Cameron','1997-12-19','Disaster/Drama',195);
INSERT INTO "Movies" VALUES(3,'Inception','Christopher Nolan','2010-07-16','Action/Science Fiction',128);
INSERT INTO "Movies" VALUES(4,'The Pursuit of Happyness','Gabriele Muccino','2006-12-15','Biography/Drama',111);
INSERT INTO "Movies" VALUES(5,'The Shawshank Redemption','Frank Darabont','1994-09-22','Drama',142);
CREATE TABLE Songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Length NUMERIC ,
    ReleaseDate DATE,
    AlbumId INTEGER,
    FOREIGN KEY(AlbumId) REFERENCES Albums(id)
);
INSERT INTO "Songs" VALUES(1,'Starlight','3:20','2019-01-01',1);
INSERT INTO "Songs" VALUES(2,'Nostalgic Night','4:15','2018-02-15',2);
INSERT INTO "Songs" VALUES(3,'Homeward Bound','4:45','2020-04-20',3);
INSERT INTO "Songs" VALUES(4,'Serenade','2:55','2017-05-25',4);
INSERT INTO "Songs" VALUES(5,'Dawn Chorus','3:30','2016-10-10',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Artists',5);
INSERT INTO "sqlite_sequence" VALUES('Songs',5);
INSERT INTO "sqlite_sequence" VALUES('MovieActors',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
INSERT INTO "sqlite_sequence" VALUES('FilmRatings',5);
INSERT INTO "sqlite_sequence" VALUES('Albums',5);
COMMIT;
