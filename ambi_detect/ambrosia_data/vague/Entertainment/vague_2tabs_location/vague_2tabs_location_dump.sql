BEGIN TRANSACTION;
CREATE TABLE Artist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city_id INTEGER,
    country_id INTEGER,
    genre TEXT,
    FOREIGN KEY(city_id) REFERENCES City(id),
    FOREIGN KEY(country_id) REFERENCES Country(id)
);
INSERT INTO "Artist" VALUES(1,'Beyonce',1,1,'Pop');
INSERT INTO "Artist" VALUES(2,'Adele',2,2,'Jazz');
INSERT INTO "Artist" VALUES(3,'The Weeknd',3,3,'R&B');
INSERT INTO "Artist" VALUES(4,'Dua Lipa',4,4,'Electronic Dance Music');
INSERT INTO "Artist" VALUES(5,'David Bowie',5,5,'Rock');
CREATE TABLE City (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    country_id INTEGER,
    FOREIGN KEY(country_id) REFERENCES Country(id)
);
INSERT INTO "City" VALUES(1,'New York',1);
INSERT INTO "City" VALUES(2,'London',2);
INSERT INTO "City" VALUES(3,'Toronto',3);
INSERT INTO "City" VALUES(4,'Melbourne',4);
INSERT INTO "City" VALUES(5,'Berlin',5);
CREATE TABLE Concert (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artist_name TEXT,
    artist_genre TEXT,
    concert_date DATE,
    city_id INTEGER,
    country_id INTEGER,
    FOREIGN KEY(artist_name) REFERENCES Artist(name),
    FOREIGN KEY(city_id) REFERENCES City(id),
    FOREIGN KEY(country_id) REFERENCES Country(id)
);
INSERT INTO "Concert" VALUES(1,'Beyonce','Pop','2022-07-01',1,1);
INSERT INTO "Concert" VALUES(2,'Adele','Jazz','2022-08-15',2,2);
INSERT INTO "Concert" VALUES(3,'The Weeknd','R&B','2022-09-01',3,3);
INSERT INTO "Concert" VALUES(4,'Dua Lipa','Electronic Dance Music','2022-10-15',4,4);
INSERT INTO "Concert" VALUES(5,'David Bowie','Rock','2022-11-01',5,5);
CREATE TABLE Country (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Country" VALUES(1,'USA');
INSERT INTO "Country" VALUES(2,'UK');
INSERT INTO "Country" VALUES(3,'Canada');
INSERT INTO "Country" VALUES(4,'Australia');
INSERT INTO "Country" VALUES(5,'Germany');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Country',5);
INSERT INTO "sqlite_sequence" VALUES('City',5);
INSERT INTO "sqlite_sequence" VALUES('Artist',5);
INSERT INTO "sqlite_sequence" VALUES('Concert',5);
COMMIT;
