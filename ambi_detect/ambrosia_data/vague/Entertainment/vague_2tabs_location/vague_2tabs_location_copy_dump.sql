BEGIN TRANSACTION;
CREATE TABLE "Artist" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city_id INTEGER,
    genre TEXT,
    FOREIGN KEY(city_id) REFERENCES "Location"(id));
INSERT INTO "Artist" VALUES(1,'Beyonce',1,'Pop');
INSERT INTO "Artist" VALUES(2,'Adele',2,'Jazz');
INSERT INTO "Artist" VALUES(3,'The Weeknd',3,'R&B');
INSERT INTO "Artist" VALUES(4,'Dua Lipa',4,'Electronic Dance Music');
INSERT INTO "Artist" VALUES(5,'David Bowie',5,'Rock');
CREATE TABLE "Concert" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artist_name TEXT,
    artist_genre TEXT,
    concert_date DATE,
    city_id INTEGER,
    FOREIGN KEY(artist_name) REFERENCES Artist(name),
    FOREIGN KEY(city_id) REFERENCES "Location"(id));
INSERT INTO "Concert" VALUES(1,'Beyonce','Pop','2022-07-01',1);
INSERT INTO "Concert" VALUES(2,'Adele','Jazz','2022-08-15',2);
INSERT INTO "Concert" VALUES(3,'The Weeknd','R&B','2022-09-01',3);
INSERT INTO "Concert" VALUES(4,'Dua Lipa','Electronic Dance Music','2022-10-15',4);
INSERT INTO "Concert" VALUES(5,'David Bowie','Rock','2022-11-01',5);
CREATE TABLE "Location" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    country_id INTEGER,
    FOREIGN KEY(country_id) REFERENCES Country(id)
);
INSERT INTO "Location" VALUES(1,'New York',1);
INSERT INTO "Location" VALUES(2,'London',2);
INSERT INTO "Location" VALUES(3,'Toronto',3);
INSERT INTO "Location" VALUES(4,'Melbourne',4);
INSERT INTO "Location" VALUES(5,'Berlin',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Location',5);
INSERT INTO "sqlite_sequence" VALUES('Artist',5);
INSERT INTO "sqlite_sequence" VALUES('Concert',5);
COMMIT;
