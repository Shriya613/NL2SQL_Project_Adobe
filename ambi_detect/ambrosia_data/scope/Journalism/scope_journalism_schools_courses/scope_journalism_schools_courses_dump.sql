BEGIN TRANSACTION;
CREATE TABLE Courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_name TEXT,
    description TEXT
);
INSERT INTO "Courses" VALUES(1,'Ethics In Journalism','This course covers ethical considerations in journalism.');
INSERT INTO "Courses" VALUES(2,'Data Analysis For Journalists','Learn how to analyze and interpret data to inform news stories.');
INSERT INTO "Courses" VALUES(3,'Feature Writing','Master the art of crafting compelling feature articles.');
INSERT INTO "Courses" VALUES(4,'News Reporting','Develop skills necessary for effective reporting on current events.');
INSERT INTO "Courses" VALUES(5,'Media Law','Understand legal issues related to media and journalism.');
CREATE TABLE Journalism_Schools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Journalism_Schools" VALUES(1,'University A','City A');
INSERT INTO "Journalism_Schools" VALUES(2,'University B','City B');
INSERT INTO "Journalism_Schools" VALUES(3,'University C','City C');
INSERT INTO "Journalism_Schools" VALUES(4,'University D','City D');
INSERT INTO "Journalism_Schools" VALUES(5,'University E','City E');
CREATE TABLE Journalism_Schools_Courses (
    journalism_schools_id INTEGER REFERENCES Journalism_Schools(id),
    course_id INTEGER REFERENCES Courses(id),
    FOREIGN KEY (journalism_schools_id) REFERENCES Journalism_Schools(id),
    FOREIGN KEY (course_id) REFERENCES Courses(id),
    PRIMARY KEY (journalism_schools_id, course_id)
);
INSERT INTO "Journalism_Schools_Courses" VALUES(1,1);
INSERT INTO "Journalism_Schools_Courses" VALUES(2,1);
INSERT INTO "Journalism_Schools_Courses" VALUES(3,1);
INSERT INTO "Journalism_Schools_Courses" VALUES(4,1);
INSERT INTO "Journalism_Schools_Courses" VALUES(5,1);
INSERT INTO "Journalism_Schools_Courses" VALUES(5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Journalism_Schools',5);
INSERT INTO "sqlite_sequence" VALUES('Courses',5);
COMMIT;
