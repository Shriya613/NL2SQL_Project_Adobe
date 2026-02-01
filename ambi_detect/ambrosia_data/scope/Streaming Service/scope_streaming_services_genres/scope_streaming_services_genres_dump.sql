BEGIN TRANSACTION;
CREATE TABLE Genres(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Genres" VALUES(1,'Drama');
INSERT INTO "Genres" VALUES(2,'Comedy');
INSERT INTO "Genres" VALUES(3,'Action');
INSERT INTO "Genres" VALUES(4,'Horror');
INSERT INTO "Genres" VALUES(5,'Documentary');
CREATE TABLE Streaming_Services(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT,
    release_date DATE);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix','2013-08-29');
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video','2006-09-07');
INSERT INTO "Streaming_Services" VALUES(3,'Hulu','2007-11-15');
INSERT INTO "Streaming_Services" VALUES(4,'Disney+','2019-11-12');
INSERT INTO "Streaming_Services" VALUES(5,'HBO Max','2020-05-27');
CREATE TABLE Streaming_Services_Genres(
    streaming_services_id INTEGER REFERENCES Streaming_Services(id),
    genres_id INTEGER REFERENCES Genres(id),
    PRIMARY KEY(streaming_services_id, genres_id));
INSERT INTO "Streaming_Services_Genres" VALUES(1,1);
INSERT INTO "Streaming_Services_Genres" VALUES(2,1);
INSERT INTO "Streaming_Services_Genres" VALUES(3,1);
INSERT INTO "Streaming_Services_Genres" VALUES(4,1);
INSERT INTO "Streaming_Services_Genres" VALUES(5,1);
INSERT INTO "Streaming_Services_Genres" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Genres',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
COMMIT;
