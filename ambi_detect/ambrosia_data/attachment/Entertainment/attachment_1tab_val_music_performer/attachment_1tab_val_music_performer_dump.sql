BEGIN TRANSACTION;
CREATE TABLE Album(
    AlbumID INTEGER PRIMARY KEY,
    Title TEXT,
    ReleaseDate DATE,
    PerformerID INTEGER,
    FOREIGN KEY (PerformerID) REFERENCES MusicPerformer(PerformerID)
);
INSERT INTO "Album" VALUES(1,'First Album','2005-03-20',1);
INSERT INTO "Album" VALUES(2,'Second Album','2007-08-15',2);
INSERT INTO "Album" VALUES(3,'Third Album','2009-11-21',3);
INSERT INTO "Album" VALUES(4,'Fourth Album','2011-06-10',4);
INSERT INTO "Album" VALUES(5,'Fifth Album','2013-09-01',5);
CREATE TABLE MovieActor(
    ActorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Nationality TEXT
);
INSERT INTO "MovieActor" VALUES(1,'John','Doe','2000-07-30','American');
INSERT INTO "MovieActor" VALUES(2,'Jane','Smith','1985-06-23','Canadian');
INSERT INTO "MovieActor" VALUES(3,'Tom','Hanks','1959-07-09','American');
INSERT INTO "MovieActor" VALUES(4,'Leonardo','DiCaprio','1974-11-11','American');
INSERT INTO "MovieActor" VALUES(5,'Scarlett','Johansson','1984-11-22','American');
CREATE TABLE MusicPerformer(
    PerformerID INTEGER PRIMARY KEY,
    Name TEXT,
    Genre TEXT,
    YearsInIndustry INT,
    MusicPerformerType TEXT
);
INSERT INTO "MusicPerformer" VALUES(1,'A Jazz Musician','Jazz',10,'Jazz Musician');
INSERT INTO "MusicPerformer" VALUES(2,'Another Jazz Musician','Jazz',5,'Jazz Musician');
INSERT INTO "MusicPerformer" VALUES(3,'A Rock Guitarist','Rock',10,'Rock Guitarist');
INSERT INTO "MusicPerformer" VALUES(4,'Another Rock Guitarist','Rock',8,'Rock Guitarist');
INSERT INTO "MusicPerformer" VALUES(5,'Pop Singer','Pop',2,'Pop Singer');
CREATE TABLE Song(
    SongID INTEGER PRIMARY KEY,
    Title TEXT,
    Length TIME,
    AlbumID INTEGER,
    FOREIGN KEY (AlbumID) REFERENCES Album(AlbumID)
);
INSERT INTO "Song" VALUES(1,'Song 1','03:00:00',1);
INSERT INTO "Song" VALUES(2,'Song 2','04:30:00',2);
INSERT INTO "Song" VALUES(3,'Song 3','03:15:00',3);
INSERT INTO "Song" VALUES(4,'Song 4','03:00:00',4);
INSERT INTO "Song" VALUES(5,'Song 5','04:15:00',5);
COMMIT;
