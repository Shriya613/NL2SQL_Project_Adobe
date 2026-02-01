BEGIN TRANSACTION;
CREATE TABLE Enrollments(
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id INTEGER,
    workshop_session_id INTEGER,
    FOREIGN KEY(participant_id) REFERENCES Participants(participant_id),
    FOREIGN KEY(workshop_session_id) REFERENCES Workshop_Sessions(session_id)
);
INSERT INTO "Enrollments" VALUES(1,1,1);
INSERT INTO "Enrollments" VALUES(2,2,2);
INSERT INTO "Enrollments" VALUES(3,3,3);
INSERT INTO "Enrollments" VALUES(4,4,4);
INSERT INTO "Enrollments" VALUES(5,5,5);
CREATE TABLE Participants(
    participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT
);
INSERT INTO "Participants" VALUES(1,'Alice','Smith','alicesmith@example.com');
INSERT INTO "Participants" VALUES(2,'Bob','Johnson','bobjohnson@example.com');
INSERT INTO "Participants" VALUES(3,'Carol','Williams','carolwilliams@example.com');
INSERT INTO "Participants" VALUES(4,'Dave','Brown','davebrown@example.com');
INSERT INTO "Participants" VALUES(5,'Eve','White','evewhite@example.com');
CREATE TABLE Photographers(
    photographer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INTEGER
);
INSERT INTO "Photographers" VALUES(1,'John Doe',8);
INSERT INTO "Photographers" VALUES(2,'Jane Smith',6);
INSERT INTO "Photographers" VALUES(3,'Michael Johnson',7);
INSERT INTO "Photographers" VALUES(4,'Emily Davis',9);
INSERT INTO "Photographers" VALUES(5,'David Brown',10);
CREATE TABLE "Photography_Workshops"(
    workshop_id INTEGER PRIMARY KEY AUTOINCREMENT,
    workshop_leader_id INTEGER,
    FOREIGN KEY(workshop_leader_id) REFERENCES Photographers(photographer_id)
);
INSERT INTO "Photography_Workshops" VALUES(1,1);
INSERT INTO "Photography_Workshops" VALUES(2,2);
INSERT INTO "Photography_Workshops" VALUES(3,3);
INSERT INTO "Photography_Workshops" VALUES(4,4);
INSERT INTO "Photography_Workshops" VALUES(5,5);
CREATE TABLE Photography_Workshops_Techniques(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photography_workshop_id INTEGER,
    technique_id INTEGER,
    FOREIGN KEY(photography_workshop_id) REFERENCES Photography_Workshops(workshop_id),
    FOREIGN KEY(technique_id) REFERENCES Techniques(technique_id)
);
INSERT INTO "Photography_Workshops_Techniques" VALUES(1,1,1);
INSERT INTO "Photography_Workshops_Techniques" VALUES(2,2,2);
INSERT INTO "Photography_Workshops_Techniques" VALUES(3,3,3);
INSERT INTO "Photography_Workshops_Techniques" VALUES(4,4,4);
INSERT INTO "Photography_Workshops_Techniques" VALUES(5,5,5);
INSERT INTO "Photography_Workshops_Techniques" VALUES(6,2,1);
INSERT INTO "Photography_Workshops_Techniques" VALUES(7,3,1);
INSERT INTO "Photography_Workshops_Techniques" VALUES(8,4,1);
INSERT INTO "Photography_Workshops_Techniques" VALUES(9,5,1);
CREATE TABLE Techniques(
    technique_id INTEGER PRIMARY KEY AUTOINCREMENT,
    technique_name TEXT
);
INSERT INTO "Techniques" VALUES(1,'Manual_Photography');
INSERT INTO "Techniques" VALUES(2,'Landscape_Photography');
INSERT INTO "Techniques" VALUES(3,'Night_Photography');
INSERT INTO "Techniques" VALUES(4,'Wedding_Photography');
INSERT INTO "Techniques" VALUES(5,'Street_Photography');
CREATE TABLE Workshop_Sessions(
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    workshop_title TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP
);
INSERT INTO "Workshop_Sessions" VALUES(1,'Introduction to Portrait Photography','2022-01-01 10:00:00','2022-01-01 16:00:00');
INSERT INTO "Workshop_Sessions" VALUES(2,'Mastering Landscape Photography','2022-01-02 10:00:00','2022-01-02 16:00:00');
INSERT INTO "Workshop_Sessions" VALUES(3,'Capturing Night Skies','2022-01-03 10:00:00','2022-01-03 16:00:00');
INSERT INTO "Workshop_Sessions" VALUES(4,'Perfecting Wedding Photography','2022-01-04 10:00:00','2022-01-04 16:00:00');
INSERT INTO "Workshop_Sessions" VALUES(5,'Urban Exploration Through Street Photography','2022-01-05 10:00:00','2022-01-05 16:00:00');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Techniques',5);
INSERT INTO "sqlite_sequence" VALUES('Photographers',5);
INSERT INTO "sqlite_sequence" VALUES('Workshop_Sessions',5);
INSERT INTO "sqlite_sequence" VALUES('Photography_Workshops_Techniques',9);
INSERT INTO "sqlite_sequence" VALUES('Participants',5);
INSERT INTO "sqlite_sequence" VALUES('Enrollments',5);
INSERT INTO "sqlite_sequence" VALUES('Photography_Workshops',5);
COMMIT;
