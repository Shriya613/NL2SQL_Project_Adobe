BEGIN TRANSACTION;
CREATE TABLE "Debut_Date" (
    id INTEGER PRIMARY KEY,
    day TEXT,
    month INTEGER
);
INSERT INTO "Debut_Date" VALUES(1,'Monday',10);
INSERT INTO "Debut_Date" VALUES(2,'Tuesday',9);
INSERT INTO "Debut_Date" VALUES(3,'Wednesday',8);
INSERT INTO "Debut_Date" VALUES(4,'Thursday',7);
INSERT INTO "Debut_Date" VALUES(5,'Friday',6);
CREATE TABLE Genre (
    id INTEGER PRIMARY KEY,
    genre_name TEXT
);
INSERT INTO "Genre" VALUES(1,'Comedy');
INSERT INTO "Genre" VALUES(2,'Drama');
INSERT INTO "Genre" VALUES(3,'Action');
INSERT INTO "Genre" VALUES(4,'Horror');
INSERT INTO "Genre" VALUES(5,'Fantasy');
CREATE TABLE "Podcast" (
    id INTEGER PRIMARY KEY,
    podcast_name TEXT,
    debut_day_id INTEGER,
    FOREIGN KEY(debut_day_id) REFERENCES "Debut_Date"(id));
INSERT INTO "Podcast" VALUES(1,'Podcast A',1);
INSERT INTO "Podcast" VALUES(2,'Podcast B',1);
INSERT INTO "Podcast" VALUES(3,'Podcast C',2);
INSERT INTO "Podcast" VALUES(4,'Podcast D',3);
INSERT INTO "Podcast" VALUES(5,'Podcast E',4);
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
