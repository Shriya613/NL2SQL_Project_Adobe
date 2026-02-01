BEGIN TRANSACTION;
CREATE TABLE Action_Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    Director TEXT);
INSERT INTO "Action_Movies" VALUES(1,'High Stakes Heist',2020,'Action','Sofia Martinez');
INSERT INTO "Action_Movies" VALUES(2,'Urban Warfare',2017,'Action','Jane Smith');
CREATE TABLE Adventure_Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    Director TEXT);
INSERT INTO "Adventure_Movies" VALUES(1,'Journey of a Lifetime',2019,'Adventure','Sofia Martinez');
INSERT INTO "Adventure_Movies" VALUES(2,'Lost Treasures',2018,'Adventure','John Doe');
CREATE TABLE Animated_Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    Director TEXT);
INSERT INTO "Animated_Movies" VALUES(1,'Whimsical Wonderland',2010,'Animated','DreamWorks Animations');
INSERT INTO "Animated_Movies" VALUES(2,'Vibrant World',2009,'Animated','Disney Pixar');
CREATE TABLE Comedy_Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    Director TEXT);
INSERT INTO "Comedy_Movies" VALUES(1,'Hilarious Mishaps',2016,'Comedy','Jack Black');
INSERT INTO "Comedy_Movies" VALUES(2,'Funny Fiasco',2015,'Comedy','Chris Rock');
CREATE TABLE Fantasy_Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    Director TEXT);
INSERT INTO "Fantasy_Movies" VALUES(1,'Enchanted Kingdom',2012,'Fantasy','Peter Jackson');
INSERT INTO "Fantasy_Movies" VALUES(2,'Magical Quest',2011,'Fantasy','Guillermo del Toro');
CREATE TABLE SciFi_Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    Director TEXT);
INSERT INTO "SciFi_Movies" VALUES(1,'Galactic Odyssey',2021,'SciFi','George Lucas');
INSERT INTO "SciFi_Movies" VALUES(2,'Interstellar Voyage',2013,'SciFi','Christopher Nolan');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Adventure_Movies',2);
INSERT INTO "sqlite_sequence" VALUES('Action_Movies',2);
INSERT INTO "sqlite_sequence" VALUES('Comedy_Movies',2);
INSERT INTO "sqlite_sequence" VALUES('SciFi_Movies',2);
INSERT INTO "sqlite_sequence" VALUES('Fantasy_Movies',2);
INSERT INTO "sqlite_sequence" VALUES('Animated_Movies',2);
COMMIT;
