BEGIN TRANSACTION;
CREATE TABLE Episodes(
    id INTEGER PRIMARY KEY,
    season_id INTEGER,
    episode_number INT,
    title TEXT,
    run_time TIME,
    airdate DATE,
    FOREIGN KEY(season_id) REFERENCES Seasons(id)
);
INSERT INTO "Episodes" VALUES(1,1,1,'Pilot','39mins','2007-09-24');
INSERT INTO "Episodes" VALUES(2,2,1,'The Pilot','22mins','1994-09-22');
INSERT INTO "Episodes" VALUES(3,3,1,'Pilot','46mins','2009-01-20');
INSERT INTO "Episodes" VALUES(4,4,1,'Winter Is Coming','55mins','2011-04-17');
CREATE TABLE Genres(
    name VARCHAR(20) PRIMARY KEY
);
INSERT INTO "Genres" VALUES('Sitcom');
INSERT INTO "Genres" VALUES('Drama');
CREATE TABLE Seasons(
    id INTEGER PRIMARY KEY,
    show_id INTEGER,
    number_of_seasons INT,
    release_year INT,
    FOREIGN KEY(show_id) REFERENCES TelevisionShows(id)
);
INSERT INTO "Seasons" VALUES(1,1,5,2008);
INSERT INTO "Seasons" VALUES(2,2,10,1994);
INSERT INTO "Seasons" VALUES(3,3,5,2009);
INSERT INTO "Seasons" VALUES(4,4,8,2011);
CREATE TABLE TelevisionShows(
    id INTEGER PRIMARY KEY,
    title TEXT,
    year INT,
    network TEXT,
    genre VARCHAR(20),
    FOREIGN KEY(genre) REFERENCES Genres(name)
);
INSERT INTO "TelevisionShows" VALUES(1,'The Big Bang Theory',2007,'CBS','Sitcom');
INSERT INTO "TelevisionShows" VALUES(2,'Friends',1994,'NBC','Sitcom');
INSERT INTO "TelevisionShows" VALUES(3,'Breaking Bad',2008,'AMC','Drama');
INSERT INTO "TelevisionShows" VALUES(4,'Game Of Thrones',2011,'HBO','Drama');
COMMIT;
