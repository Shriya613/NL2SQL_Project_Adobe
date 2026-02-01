BEGIN TRANSACTION;
CREATE TABLE Actor(
    ActorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    RoleInFilm ID INTEGER,
    FilmID INTEGER,
    FOREIGN KEY (RoleInFilm) REFERENCES Film(FilmID),
    FOREIGN KEY (FilmID) REFERENCES Film(FilmID)
);
INSERT INTO "Actor" VALUES(1,'Leonardo','DiCaprio','1974-11-11',1,1);
INSERT INTO "Actor" VALUES(2,'Christian','Bale','1974-01-30',2,2);
INSERT INTO "Actor" VALUES(3,'Kate','Winslet','1975-10-05',3,1);
INSERT INTO "Actor" VALUES(4,'Samuel','L. Jackson','1948-12-21',4,3);
INSERT INTO "Actor" VALUES(5,'Tom','Hanks','1956-07-09',5,4);
CREATE TABLE Director(
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    DirectedFilm ID INTEGER,
    FilmID INTEGER,
    FOREIGN KEY (DirectedFilm) REFERENCES Film(FilmID),
    FOREIGN KEY (FilmID) REFERENCES Film(FilmID)
);
INSERT INTO "Director" VALUES(1,'Christopher','Nolan','1970-07-30',1,1);
INSERT INTO "Director" VALUES(2,'James','Cameron','1954-08-16',2,2);
INSERT INTO "Director" VALUES(3,'Robert','Zemeckis','1951-05-14',3,1);
INSERT INTO "Director" VALUES(4,'Steven','Spielberg','1946-12-18',4,3);
INSERT INTO "Director" VALUES(5,'Ron','Howard','1954-03-13',5,4);
CREATE TABLE FightScenes(
    SceneID INTEGER PRIMARY KEY AUTOINCREMENT,
    Choreographer TEXT,
    Experience TEXT,
    SceneDescription TEXT,
    FilmID INTEGER,
    FOREIGN KEY (FilmID) REFERENCES Film(FilmID)
);
INSERT INTO "FightScenes" VALUES(1,'Jackie','10 years','A fight scene on a train.',1);
INSERT INTO "FightScenes" VALUES(2,'Alex','5 years','A fight scene at an airport.',2);
INSERT INTO "FightScenes" VALUES(3,'Luke','15 years','A fight scene in a dark alley.',3);
INSERT INTO "FightScenes" VALUES(4,'Helen','20 years','A fight scene in a castle.',4);
INSERT INTO "FightScenes" VALUES(5,'Mike','7 years','A fight scene on a rooftop.',5);
CREATE TABLE Film(
    FilmID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INTEGER,
    Budget REAL,
    BoxOffice REAL,
    HouseID INTEGER,
    FOREIGN KEY (HouseID) REFERENCES ProductionHouse(HouseID)
);
INSERT INTO "Film" VALUES(1,'The Dark Knight',2008,180000000.0,1006000000.0,1);
INSERT INTO "Film" VALUES(2,'Avatar',2009,300000000.0,2748000000.0,2);
INSERT INTO "Film" VALUES(3,'Titanic',1997,200000000.0,2202000000.0,3);
INSERT INTO "Film" VALUES(4,'Jurassic Park',1993,63000000.0,914000000.0,4);
INSERT INTO "Film" VALUES(5,'Forrest Gump',1994,55000000.0,677000000.0,5);
CREATE TABLE Producer(
    ProducerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE,
    ProducedFilm ID INTEGER,
    FilmID INTEGER,
    FOREIGN KEY (ProducedFilm) REFERENCES Film(FilmID),
    FOREIGN KEY (FilmID) REFERENCES Film(FilmID)
);
INSERT INTO "Producer" VALUES(1,'Debra','Franicich','1957-07-01',1,1);
INSERT INTO "Producer" VALUES(2,'Joel','Silva','1965-08-05',2,2);
INSERT INTO "Producer" VALUES(3,'Bonnie','Timmermann','1963-06-12',3,1);
INSERT INTO "Producer" VALUES(4,'Michael','De Luca','1961-07-13',4,3);
INSERT INTO "Producer" VALUES(5,'Gary','Goetzman','1960-08-16',5,4);
CREATE TABLE ProductionHouse(
    HouseID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "ProductionHouse" VALUES(1,'New Horizon Pictures');
INSERT INTO "ProductionHouse" VALUES(2,'Star Studios');
INSERT INTO "ProductionHouse" VALUES(3,'Silver Screen Entertainment');
INSERT INTO "ProductionHouse" VALUES(4,'Moonlight Films');
INSERT INTO "ProductionHouse" VALUES(5,'Sunset Cinema');
CREATE TABLE Stunts(
    StuntID INTEGER PRIMARY KEY AUTOINCREMENT,
    Coordinator TEXT,
    Experience TEXT,
    SceneID INTEGER,
    FilmID INTEGER,
    FOREIGN KEY (SceneID) REFERENCES Film(FilmID),
    FOREIGN KEY (FilmID) REFERENCES Film(FilmID)
);
INSERT INTO "Stunts" VALUES(1,'Johnny','10 years',NULL,1);
INSERT INTO "Stunts" VALUES(2,'Emily','5 years',NULL,2);
INSERT INTO "Stunts" VALUES(3,'David','15 years',NULL,3);
INSERT INTO "Stunts" VALUES(4,'Anna','20 years',NULL,4);
INSERT INTO "Stunts" VALUES(5,'Mark','7 years',NULL,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('ProductionHouse',5);
INSERT INTO "sqlite_sequence" VALUES('Film',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Producer',5);
INSERT INTO "sqlite_sequence" VALUES('Stunts',5);
INSERT INTO "sqlite_sequence" VALUES('FightScenes',5);
COMMIT;
