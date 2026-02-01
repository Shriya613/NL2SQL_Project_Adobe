BEGIN TRANSACTION;
CREATE TABLE Genres(
    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    genre_name TEXT
);
INSERT INTO "Genres" VALUES(1,'Rock_Music');
INSERT INTO "Genres" VALUES(2,'Pop_Music');
INSERT INTO "Genres" VALUES(3,'Jazz');
INSERT INTO "Genres" VALUES(4,'Country');
INSERT INTO "Genres" VALUES(5,'Electronic');
CREATE TABLE Music_Festivals(
    festival_id INTEGER PRIMARY KEY AUTOINCREMENT,
    festival_name TEXT,
    location TEXT
);
INSERT INTO "Music_Festivals" VALUES(1,'Coachella','Indio, California');
INSERT INTO "Music_Festivals" VALUES(2,'Glastonbury','Pilton, United Kingdom');
INSERT INTO "Music_Festivals" VALUES(3,'Lollapalooza','Chicago, Illinois');
INSERT INTO "Music_Festivals" VALUES(4,'Burning Man','Black Rock City, Nevada');
INSERT INTO "Music_Festivals" VALUES(5,'Tomorrowland','Boom, Belgium');
CREATE TABLE Music_Festivals_Genres(
    music_festival_id INTEGER,
    genre_id INTEGER,
    FOREIGN KEY (music_festival_id) REFERENCES Music_Festivals(festival_id),
    FOREIGN KEY (genre_id) REFERENCES Genres(genre_id)
);
INSERT INTO "Music_Festivals_Genres" VALUES(1,1);
INSERT INTO "Music_Festivals_Genres" VALUES(1,2);
INSERT INTO "Music_Festivals_Genres" VALUES(2,3);
INSERT INTO "Music_Festivals_Genres" VALUES(3,4);
INSERT INTO "Music_Festivals_Genres" VALUES(3,5);
INSERT INTO "Music_Festivals_Genres" VALUES(4,1);
INSERT INTO "Music_Festivals_Genres" VALUES(4,2);
INSERT INTO "Music_Festivals_Genres" VALUES(5,3);
INSERT INTO "Music_Festivals_Genres" VALUES(5,4);
INSERT INTO "Music_Festivals_Genres" VALUES(2,1);
INSERT INTO "Music_Festivals_Genres" VALUES(3,1);
INSERT INTO "Music_Festivals_Genres" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Genres',5);
INSERT INTO "sqlite_sequence" VALUES('Music_Festivals',5);
COMMIT;
