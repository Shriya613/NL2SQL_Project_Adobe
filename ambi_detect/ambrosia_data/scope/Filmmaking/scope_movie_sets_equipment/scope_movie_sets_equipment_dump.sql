BEGIN TRANSACTION;
CREATE TABLE Equipment(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    type TEXT );
INSERT INTO "Equipment" VALUES(1,'Canon C200','Camera');
INSERT INTO "Equipment" VALUES(2,'Arri Alexa Mini','Camera');
INSERT INTO "Equipment" VALUES(3,'DJI Ronin-SC','Gimbal');
INSERT INTO "Equipment" VALUES(4,'Rokinon Xeen 35mm T1.5','Lens');
INSERT INTO "Equipment" VALUES(5,'Mathews Freefly MOVI M10','Crane');
CREATE TABLE Movie_sets(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    director TEXT,
    release_date DATE);
INSERT INTO "Movie_sets" VALUES(1,'The Dark Knight','Christopher Nolan','2008-07-16');
INSERT INTO "Movie_sets" VALUES(2,'Inception','Christopher Nolan','2010-07-16');
INSERT INTO "Movie_sets" VALUES(3,'Interstellar','Christopher Nolan','2014-11-07');
INSERT INTO "Movie_sets" VALUES(4,'Mad Max: Fury Road','George Miller','2015-05-14');
INSERT INTO "Movie_sets" VALUES(5,'Birdman','Alejandro González Iñárritu','2014-10-16');
CREATE TABLE Movie_sets_equipment(
    movie_set_id INTEGER,
    equipment_id INTEGER,
    quantity INTEGER DEFAULT 1,
    FOREIGN KEY (movie_set_id) REFERENCES Movie_sets(id),
    FOREIGN KEY (equipment_id) REFERENCES Equipment(id));
INSERT INTO "Movie_sets_equipment" VALUES(1,1,2);
INSERT INTO "Movie_sets_equipment" VALUES(1,2,3);
INSERT INTO "Movie_sets_equipment" VALUES(2,1,4);
INSERT INTO "Movie_sets_equipment" VALUES(2,3,2);
INSERT INTO "Movie_sets_equipment" VALUES(3,1,1);
INSERT INTO "Movie_sets_equipment" VALUES(3,4,1);
INSERT INTO "Movie_sets_equipment" VALUES(4,2,1);
INSERT INTO "Movie_sets_equipment" VALUES(4,3,1);
INSERT INTO "Movie_sets_equipment" VALUES(4,1,1);
INSERT INTO "Movie_sets_equipment" VALUES(5,1,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Equipment',5);
INSERT INTO "sqlite_sequence" VALUES('Movie_sets',5);
COMMIT;
