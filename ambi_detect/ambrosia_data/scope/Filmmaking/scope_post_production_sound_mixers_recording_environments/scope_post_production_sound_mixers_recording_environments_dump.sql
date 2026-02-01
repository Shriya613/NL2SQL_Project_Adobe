BEGIN TRANSACTION;
CREATE TABLE FilmProjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE
);
INSERT INTO "FilmProjects" VALUES(1,'Film A','2023-01-01');
INSERT INTO "FilmProjects" VALUES(2,'Film B','2023-02-01');
INSERT INTO "FilmProjects" VALUES(3,'Film C','2023-03-01');
INSERT INTO "FilmProjects" VALUES(4,'Film D','2023-04-01');
INSERT INTO "FilmProjects" VALUES(5,'Film E','2023-05-01');
CREATE TABLE Films_with_Sound_Mixing (
    film_id INTEGER,
    mixer_id INTEGER,
    FOREIGN KEY (film_id) REFERENCES FilmProjects (id),
    FOREIGN KEY (mixer_id) REFERENCES Post_Production_Sound_Mixers (id),
    PRIMARY KEY (film_id, mixer_id)
);
INSERT INTO "Films_with_Sound_Mixing" VALUES(1,1);
INSERT INTO "Films_with_Sound_Mixing" VALUES(2,2);
INSERT INTO "Films_with_Sound_Mixing" VALUES(3,3);
INSERT INTO "Films_with_Sound_Mixing" VALUES(4,4);
INSERT INTO "Films_with_Sound_Mixing" VALUES(5,5);
CREATE TABLE Post_Production_Sound_Mixers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    years_of_experience INTEGER
);
INSERT INTO "Post_Production_Sound_Mixers" VALUES(1,'John Doe',10);
INSERT INTO "Post_Production_Sound_Mixers" VALUES(2,'Jane Smith',8);
INSERT INTO "Post_Production_Sound_Mixers" VALUES(3,'Alice Johnson',7);
INSERT INTO "Post_Production_Sound_Mixers" VALUES(4,'Bob Brown',6);
INSERT INTO "Post_Production_Sound_Mixers" VALUES(5,'Tom Green',9);
CREATE TABLE Recording_Environments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    environment_name TEXT 
);
INSERT INTO "Recording_Environments" VALUES(1,'Isolation Booths');
INSERT INTO "Recording_Environments" VALUES(2,'Live Studio');
INSERT INTO "Recording_Environments" VALUES(3,'Outdoor Location');
INSERT INTO "Recording_Environments" VALUES(4,'Foley Stage');
INSERT INTO "Recording_Environments" VALUES(5,'Dubbing Theatre');
CREATE TABLE Sound_Mixer_Environment_History (
    sound_mixer_id INTEGER,
    environment_id INTEGER,
    start_year INTEGER,
    end_year INTEGER,
    FOREIGN KEY (sound_mixer_id) REFERENCES Post_Production_Sound_Mixers (id),
    FOREIGN KEY (environment_id) REFERENCES Recording_Environments (id)
);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(1,1,2013,2017);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(1,2,2017,2020);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(2,2,2015,2019);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(2,3,2019,2022);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(3,3,2014,2018);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(2,1,2015,2019);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(3,1,2014,2018);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(4,1,NULL,NULL);
INSERT INTO "Sound_Mixer_Environment_History" VALUES(5,1,NULL,NULL);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Recording_Environments',5);
INSERT INTO "sqlite_sequence" VALUES('Post_Production_Sound_Mixers',5);
INSERT INTO "sqlite_sequence" VALUES('FilmProjects',5);
COMMIT;
