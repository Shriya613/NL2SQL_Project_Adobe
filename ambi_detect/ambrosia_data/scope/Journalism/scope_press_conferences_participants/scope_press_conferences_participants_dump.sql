BEGIN TRANSACTION;
CREATE TABLE Participants(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT ,
    name TEXT);
INSERT INTO "Participants" VALUES(1,'Reporter','John Smith');
INSERT INTO "Participants" VALUES(2,'Reporter','Jane Doe');
INSERT INTO "Participants" VALUES(3,'Spokesperson','Mike Johnson');
INSERT INTO "Participants" VALUES(4,'Expert Panelist','Emily Davis');
INSERT INTO "Participants" VALUES(5,'Head of Department','David Brown');
CREATE TABLE Press_Conferences(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date DATE);
INSERT INTO "Press_Conferences" VALUES(1,'Presidential Speech','2023-01-01');
INSERT INTO "Press_Conferences" VALUES(2,'Tech Company Announcement','2023-01-02');
INSERT INTO "Press_Conferences" VALUES(3,'Economic Summit','2023-01-03');
INSERT INTO "Press_Conferences" VALUES(4,'Healthcare Conference','2023-01-04');
INSERT INTO "Press_Conferences" VALUES(5,'Environmental Symposium','2023-01-05');
CREATE TABLE Press_Conferences_Participants(
    press_conference_id INTEGER,
    participant_id INTEGER,
    FOREIGN KEY (press_conference_id) REFERENCES Press_Conferences(id),
    FOREIGN KEY (participant_id) REFERENCES Participants(id));
INSERT INTO "Press_Conferences_Participants" VALUES(1,1);
INSERT INTO "Press_Conferences_Participants" VALUES(2,1);
INSERT INTO "Press_Conferences_Participants" VALUES(3,1);
INSERT INTO "Press_Conferences_Participants" VALUES(4,1);
INSERT INTO "Press_Conferences_Participants" VALUES(5,1);
INSERT INTO "Press_Conferences_Participants" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Press_Conferences',5);
INSERT INTO "sqlite_sequence" VALUES('Participants',5);
COMMIT;
