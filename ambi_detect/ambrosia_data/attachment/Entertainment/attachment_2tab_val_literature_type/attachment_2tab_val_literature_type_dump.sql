BEGIN TRANSACTION;
CREATE TABLE Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_title TEXT,
    director TEXT,
    release_year INTEGER,
    genre TEXT,
    rating REAL  );
INSERT INTO "Movies" VALUES(1,'La Dolce Vita','Federico Fellini',1960,'Drama',8.5);
INSERT INTO "Movies" VALUES(2,'Pulp Fiction','Quentin Tarantino',1994,'Crime',8.9);
CREATE TABLE MusicAlbums(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    album_name TEXT,
    artist TEXT,
    release_date DATE,
    genre TEXT,
    number_of_tracks INTEGER );
INSERT INTO "MusicAlbums" VALUES(1,'Dark Side of the Moon','Pink Floyd','1973-03-24','Progressive Rock',43);
INSERT INTO "MusicAlbums" VALUES(2,'Abbey Road','The Beatles','1969-09-26','Rock',17);
CREATE TABLE Novels(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    author TEXT,
    publication_year INTEGER,
    genre TEXT,
    language TEXT);
INSERT INTO "Novels" VALUES(1,'Il Canto','Umberto Eco',1980,'Historical Fiction','Italian');
INSERT INTO "Novels" VALUES(2,'To Kill a Mockingbird','Harper Lee',1960,'Classic','English');
CREATE TABLE Poems(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    poem_name TEXT,
    poet TEXT,
    published_date DATE,
    theme TEXT,
    language TEXT);
INSERT INTO "Poems" VALUES(1,'The Waste Land','T. S. Eliot','1922-06-30','War & Disillusionment','English');
INSERT INTO "Poems" VALUES(2,'Dante Alighieri','Dante Alighieri','circa 1307-1321','Spiritual Journey','Italian');
CREATE TABLE TVShows(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    show_name TEXT,
    creator TEXT,
    start_year INTEGER,
    end_year INTEGER,
    genre TEXT);
INSERT INTO "TVShows" VALUES(1,'Breaking Bad','Vince Gilligan',2008,2013,'Crime Drama');
INSERT INTO "TVShows" VALUES(2,'Game of Thrones','David Benioff, D. B. Weiss',2011,2019,'Fantasy');
CREATE TABLE VideoGames(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_name TEXT,
    developer TEXT,
    release_date DATE,
    genre TEXT,
    platform TEXT);
INSERT INTO "VideoGames" VALUES(1,'Red Dead Redemption 2','Rockstar Studios','2018-10-26','Action Adventure','PlayStation 4');
INSERT INTO "VideoGames" VALUES(2,'God of War','Santa Monica Studio','2018-04-20','Action Adventure','PlayStation 4');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Novels',2);
INSERT INTO "sqlite_sequence" VALUES('Poems',2);
INSERT INTO "sqlite_sequence" VALUES('Movies',2);
INSERT INTO "sqlite_sequence" VALUES('MusicAlbums',2);
INSERT INTO "sqlite_sequence" VALUES('VideoGames',2);
INSERT INTO "sqlite_sequence" VALUES('TVShows',2);
COMMIT;
