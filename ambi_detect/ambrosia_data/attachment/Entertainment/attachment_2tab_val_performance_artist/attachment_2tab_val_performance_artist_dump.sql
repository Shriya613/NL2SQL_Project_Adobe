BEGIN TRANSACTION;
CREATE TABLE games(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    platform TEXT,
    genre TEXT,
    developer TEXT
);
INSERT INTO "games" VALUES(1,'Red Dead Redemption 2','PlayStation 4','Western Action-adventure','Rockstar Studios');
INSERT INTO "games" VALUES(2,'Call Of Duty: Modern Warfare','Xbox One','First Person Shooter','Activision');
INSERT INTO "games" VALUES(3,'Assassin’s Creed Odyssey','PC','Action RPG','Ubisoft Quebec');
INSERT INTO "games" VALUES(4,'Fortnite Battle Royale','iOS','Battle Royale','Epic Games');
INSERT INTO "games" VALUES(5,'Overwatch','Windows','Team-based Multiplayer First-person shooter','Blizzard Entertainment');
CREATE TABLE magicians(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    duration_of_performance INTEGER,
    venue_type TEXT,
    show_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "magicians" VALUES(1,'David Copperfield',60,'Las Vegas Hotel','2024-03-13 18:23:49');
INSERT INTO "magicians" VALUES(2,'Penn & Teller',75,'Rio All Suite Hotel & Casino','2024-03-13 18:23:49');
INSERT INTO "magicians" VALUES(3,'Criss Angel',90,'Planet Hollywood Resort & Casino','2024-03-13 18:23:49');
INSERT INTO "magicians" VALUES(4,'Derren Brown',65,'London West End','2024-03-13 18:23:49');
INSERT INTO "magicians" VALUES(5,'David Blaine',80,'New York City Streets','2024-03-13 18:23:49');
CREATE TABLE movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    director TEXT
);
INSERT INTO "movies" VALUES(1,'Inception',2010,'Action/Adventure','Christopher Nolan');
INSERT INTO "movies" VALUES(2,'Avatar',2009,'Science Fiction','James Cameron');
INSERT INTO "movies" VALUES(3,'Interstellar',2014,'Science Fiction','Christopher Nolan');
INSERT INTO "movies" VALUES(4,'Gravity',2013,'Thriller','Alfonso Cuaron');
INSERT INTO "movies" VALUES(5,'Titanic',1997,'Romance','James Cameron');
CREATE TABLE musicalbums(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artist TEXT,
    albumname TEXT,
    genre TEXT,
    releaseyear INTEGER
);
INSERT INTO "musicalbums" VALUES(1,'Adele','21','Pop',2011);
INSERT INTO "musicalbums" VALUES(2,'Beyonce','Lemonade','Contemporary R&B',2016);
INSERT INTO "musicalbums" VALUES(3,'Ed Sheeran','÷','Pop',2017);
INSERT INTO "musicalbums" VALUES(4,'Sam Smith','The Thrill Of It All','Pop',2017);
INSERT INTO "musicalbums" VALUES(5,'Justin Timberlake','Man Of The Woods','Pop',2018);
CREATE TABLE standupcomedians(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    duration_of_performance INTEGER,
    performance_venue TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "standupcomedians" VALUES(1,'Jerry Seinfeld',60,'The Beacon Theatre','2024-03-13 18:23:49');
INSERT INTO "standupcomedians" VALUES(2,'Chris Rock',90,'Madison Square Garden','2024-03-13 18:23:49');
INSERT INTO "standupcomedians" VALUES(3,'Kevin Hart',75,'Hollywood Bowl','2024-03-13 18:23:49');
INSERT INTO "standupcomedians" VALUES(4,'Dave Chappelle',80,'Radio City Music Hall','2024-03-13 18:23:49');
INSERT INTO "standupcomedians" VALUES(5,'Bill Burr',65,'Tower Theater','2024-03-13 18:23:49');
CREATE TABLE tvshows(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    season INTEGER,
    episode INTEGER,
    network TEXT
);
INSERT INTO "tvshows" VALUES(1,'Breaking Bad',5,16,'AMC Network');
INSERT INTO "tvshows" VALUES(2,'Game Of Thrones',8,60,'HBO');
INSERT INTO "tvshows" VALUES(3,'Stranger Things',3,8,'Netflix');
INSERT INTO "tvshows" VALUES(4,'Friends',10,236,'NBC');
INSERT INTO "tvshows" VALUES(5,'The Office US',9,201,'NBC');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('standupcomedians',5);
INSERT INTO "sqlite_sequence" VALUES('magicians',5);
INSERT INTO "sqlite_sequence" VALUES('movies',5);
INSERT INTO "sqlite_sequence" VALUES('tvshows',5);
INSERT INTO "sqlite_sequence" VALUES('games',5);
INSERT INTO "sqlite_sequence" VALUES('musicalbums',5);
COMMIT;
