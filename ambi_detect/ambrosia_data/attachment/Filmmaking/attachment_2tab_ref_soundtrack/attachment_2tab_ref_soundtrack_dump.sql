BEGIN TRANSACTION;
CREATE TABLE Creators(
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE);
INSERT INTO "Creators" VALUES(1,'John Williams');
INSERT INTO "Creators" VALUES(2,'Hans Zimmer');
INSERT INTO "Creators" VALUES(3,'James Horner');
INSERT INTO "Creators" VALUES(4,'Danny Elfman');
INSERT INTO "Creators" VALUES(5,'Thomas Newman');
CREATE TABLE Films(
    id INTEGER PRIMARY KEY,
    title TEXT,
    releaseDate DATE,
    soundDesign INTEGER,
    musicScore INTEGER,
    FOREIGN KEY(soundDesign) REFERENCES SoundDesigns(id),
    FOREIGN KEY(musicScore) REFERENCES MusicScores(id));
INSERT INTO "Films" VALUES(1,'Star Wars','1977-05-25',1,3);
INSERT INTO "Films" VALUES(2,'Inception','2010-07-16',2,4);
INSERT INTO "Films" VALUES(3,'Titanic','1997-12-18',3,5);
INSERT INTO "Films" VALUES(4,'Batman','1989-06-23',4,6);
INSERT INTO "Films" VALUES(5,'The Nightmare Before Christmas','1993-10-08',5,7);
CREATE TABLE MusicScores(
    id INTEGER PRIMARY KEY,
    description TEXT,
    creatorId INTEGER,
    filmId INTEGER,
    FOREIGN KEY(creatorId) REFERENCES Creators(id),
    FOREIGN KEY(filmId) REFERENCES Films(id));
INSERT INTO "MusicScores" VALUES(1,'Orchestral Epic Theme',1,1);
INSERT INTO "MusicScores" VALUES(2,'Modern Electronic Score',2,2);
INSERT INTO "MusicScores" VALUES(3,'Romantic Love Theme',3,3);
INSERT INTO "MusicScores" VALUES(4,'Mysterious Adventure Motif',4,4);
INSERT INTO "MusicScores" VALUES(5,'Whimsical Halloween Melody',5,5);
CREATE TABLE SoundDesigns(
    id INTEGER PRIMARY KEY,
    description TEXT,
    creatorId INTEGER,
    filmId INTEGER,
    FOREIGN KEY(creatorId) REFERENCES Creators(id),
    FOREIGN KEY(filmId) REFERENCES Films(id));
INSERT INTO "SoundDesigns" VALUES(1,'Epic Space Battle Sounds',1,1);
INSERT INTO "SoundDesigns" VALUES(2,'Complex Dream Sequence Sounds',2,2);
INSERT INTO "SoundDesigns" VALUES(3,'Giant Ship Sinking Sounds',3,3);
INSERT INTO "SoundDesigns" VALUES(4,'Dark and Mysterious Atmosphere Sounds',4,4);
INSERT INTO "SoundDesigns" VALUES(5,'Spooky Town Sounds',5,5);
COMMIT;
