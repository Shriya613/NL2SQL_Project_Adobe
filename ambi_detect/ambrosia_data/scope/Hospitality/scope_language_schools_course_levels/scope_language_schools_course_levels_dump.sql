BEGIN TRANSACTION;
CREATE TABLE Course_Levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT
);
INSERT INTO "Course_Levels" VALUES(1,'Beginner_Level');
INSERT INTO "Course_Levels" VALUES(2,'Intermediate_Level');
INSERT INTO "Course_Levels" VALUES(3,'Advanced_Level');
INSERT INTO "Course_Levels" VALUES(4,'Expert_Level');
CREATE TABLE Language_Schools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Language_Schools" VALUES(1,'ABC School','New York');
INSERT INTO "Language_Schools" VALUES(2,'DEF School','London');
INSERT INTO "Language_Schools" VALUES(3,'GHI School','Paris');
INSERT INTO "Language_Schools" VALUES(4,'JKL School','Berlin');
INSERT INTO "Language_Schools" VALUES(5,'MNO School','Rome');
CREATE TABLE Language_Schools_Course_Levels (
    language_schools_id INTEGER,
    course_levels_id INTEGER,
    FOREIGN KEY (language_schools_id) REFERENCES Language_Schools (id),
    FOREIGN KEY (course_levels_id) REFERENCES Course_Levels (id),
    PRIMARY KEY (language_schools_id, course_levels_id)
);
INSERT INTO "Language_Schools_Course_Levels" VALUES(2,1);
INSERT INTO "Language_Schools_Course_Levels" VALUES(3,1);
INSERT INTO "Language_Schools_Course_Levels" VALUES(4,1);
INSERT INTO "Language_Schools_Course_Levels" VALUES(5,1);
INSERT INTO "Language_Schools_Course_Levels" VALUES(1,2);
INSERT INTO "Language_Schools_Course_Levels" VALUES(3,2);
INSERT INTO "Language_Schools_Course_Levels" VALUES(4,2);
INSERT INTO "Language_Schools_Course_Levels" VALUES(5,2);
INSERT INTO "Language_Schools_Course_Levels" VALUES(1,3);
INSERT INTO "Language_Schools_Course_Levels" VALUES(2,3);
INSERT INTO "Language_Schools_Course_Levels" VALUES(4,3);
INSERT INTO "Language_Schools_Course_Levels" VALUES(5,3);
INSERT INTO "Language_Schools_Course_Levels" VALUES(1,4);
INSERT INTO "Language_Schools_Course_Levels" VALUES(2,4);
INSERT INTO "Language_Schools_Course_Levels" VALUES(3,4);
INSERT INTO "Language_Schools_Course_Levels" VALUES(4,4);
INSERT INTO "Language_Schools_Course_Levels" VALUES(5,4);
INSERT INTO "Language_Schools_Course_Levels" VALUES(1,1);
INSERT INTO "Language_Schools_Course_Levels" VALUES(2,2);
INSERT INTO "Language_Schools_Course_Levels" VALUES(3,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Language_Schools',5);
INSERT INTO "sqlite_sequence" VALUES('Course_Levels',4);
COMMIT;
