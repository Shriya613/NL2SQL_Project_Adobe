BEGIN TRANSACTION;
CREATE TABLE Debut_Day (
    id INTEGER PRIMARY KEY,
    day TEXT,
    month INTEGER
);
INSERT INTO "Debut_Day" VALUES(1,'Monday',10);
INSERT INTO "Debut_Day" VALUES(2,'Tuesday',9);
INSERT INTO "Debut_Day" VALUES(3,'Wednesday',8);
INSERT INTO "Debut_Day" VALUES(4,'Thursday',7);
INSERT INTO "Debut_Day" VALUES(5,'Friday',6);
CREATE TABLE Debut_Year (
    id INTEGER PRIMARY KEY,
    year INTEGER
);
INSERT INTO "Debut_Year" VALUES(1,2019);
INSERT INTO "Debut_Year" VALUES(2,2020);
INSERT INTO "Debut_Year" VALUES(3,2021);
INSERT INTO "Debut_Year" VALUES(4,2022);
INSERT INTO "Debut_Year" VALUES(5,2023);
CREATE TABLE Genre (
    id INTEGER PRIMARY KEY,
    genre_name TEXT
);
INSERT INTO "Genre" VALUES(1,'Comedy');
INSERT INTO "Genre" VALUES(2,'Drama');
INSERT INTO "Genre" VALUES(3,'Action');
INSERT INTO "Genre" VALUES(4,'Horror');
INSERT INTO "Genre" VALUES(5,'Fantasy');
CREATE TABLE Podcast (
    id INTEGER PRIMARY KEY,
    podcast_name TEXT,
    debut_day_id INTEGER,
    debut_year_id INTEGER,
    FOREIGN KEY(debut_day_id) REFERENCES Debut_Day(id),
    FOREIGN KEY(debut_year_id) REFERENCES Debut_Year(id)
);
INSERT INTO "Podcast" VALUES(1,'Podcast A',1,1);
INSERT INTO "Podcast" VALUES(2,'Podcast B',1,2);
INSERT INTO "Podcast" VALUES(3,'Podcast C',2,3);
INSERT INTO "Podcast" VALUES(4,'Podcast D',3,4);
INSERT INTO "Podcast" VALUES(5,'Podcast E',4,5);
CREATE TABLE Podcast_Genre (
    pod_genre_id INTEGER PRIMARY KEY,
    podcast_id INTEGER,
    genre_id INTEGER,
    FOREIGN KEY(podcast_id) REFERENCES Podcast(id),
    FOREIGN KEY(genre_id) REFERENCES Genre(id)
);
INSERT INTO "Podcast_Genre" VALUES(1,1,1);
INSERT INTO "Podcast_Genre" VALUES(2,2,2);
INSERT INTO "Podcast_Genre" VALUES(3,3,3);
INSERT INTO "Podcast_Genre" VALUES(4,4,4);
INSERT INTO "Podcast_Genre" VALUES(5,5,5);
COMMIT;
