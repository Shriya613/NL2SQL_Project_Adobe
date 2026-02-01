BEGIN TRANSACTION;
CREATE TABLE Actor(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Birthdate DATE,
    Nationality TEXT,
    RoleInFilm TEXT
);
INSERT INTO "Actor" VALUES(1,'Morgan Freeman','1937-06-01','American','Red');
INSERT INTO "Actor" VALUES(2,'Brad Pitt','1963-12-18','American','James');
CREATE TABLE Cinematography(
    ID INTEGER PRIMARY KEY,
    Technique TEXT,
    Equipment TEXT,
    ShotList TEXT,
    FilmID INTEGER,
    FOREIGN KEY(FilmID) REFERENCES Film(ID)
);
INSERT INTO "Cinematography" VALUES(1,'Handheld','ARRI Alexa Mini','Scene 1: Establishing shot of prison...',1);
INSERT INTO "Cinematography" VALUES(2,'Dolly Zoom','Panavision Millennium XL2','Scene 3: Close up on Andy and Red talking...',1);
CREATE TABLE Film(
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Director TEXT,
    ReleaseDate DATE,
    Genre TEXT,
    Rating REAL 
);
INSERT INTO "Film" VALUES(1,'The Shawshank Redemption','Frank Darabont','1994-09-23','Drama',8.7);
INSERT INTO "Film" VALUES(2,'Pulp Fiction','Quentin Tarantino','1994-10-14','Crime',8.9);
CREATE TABLE ProductionDesign(
    ID INTEGER PRIMARY KEY,
    Style TEXT,
    Budget REAL,
    Location TEXT,
    FilmID INTEGER,
    FOREIGN KEY(FilmID) REFERENCES Film(ID)
);
INSERT INTO "ProductionDesign" VALUES(1,'Minimalist','10 million','Prison in Ohio',1);
INSERT INTO "ProductionDesign" VALUES(2,'Neo-noir','5 million','Los Angeles streets',2);
CREATE TABLE "Reviews"(
    ID INTEGER PRIMARY KEY,
    CriticName TEXT,
    Review TEXT,
    FilmId INTEGER,
    FOREIGN KEY(FilmId) REFERENCES Film(ID)
);
INSERT INTO "Reviews" VALUES(1,'Roger Ebert','A masterpiece.',1);
INSERT INTO "Reviews" VALUES(2,'Pauline Kael','Terrific acting.',2);
CREATE TABLE SoundDesign(
    ID INTEGER PRIMARY KEY,
    Type TEXT,
    Effects TEXT,
    MixingLocation TEXT,
    FilmID INTEGER,
    FOREIGN KEY(FilmID) REFERENCES Film(ID)
);
INSERT INTO "SoundDesign" VALUES(1,'Dialogue and Music','Gunshots, screams','Dolby Theatre',1);
INSERT INTO "SoundDesign" VALUES(2,'Sound effects','Car screeching, footsteps','Dolby Theatre',2);
COMMIT;
