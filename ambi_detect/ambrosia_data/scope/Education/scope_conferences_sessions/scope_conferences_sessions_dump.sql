BEGIN TRANSACTION;
CREATE TABLE Conference_Sessions (
    conference_session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conference_id INTEGER,
    session_type_id INTEGER,
    FOREIGN KEY (conference_id) REFERENCES Conferences(conference_id),
    FOREIGN KEY (session_type_id) REFERENCES Sessions(session_type_id)
);
INSERT INTO "Conference_Sessions" VALUES(1,1,1);
INSERT INTO "Conference_Sessions" VALUES(2,2,1);
INSERT INTO "Conference_Sessions" VALUES(3,3,1);
INSERT INTO "Conference_Sessions" VALUES(4,4,1);
INSERT INTO "Conference_Sessions" VALUES(5,5,1);
CREATE TABLE Conference_Speakers (
    conference_speaker_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conference_id INTEGER,
    speaker_id INTEGER,
    FOREIGN KEY (conference_id) REFERENCES Conferences(conference_id),
    FOREIGN KEY (speaker_id) REFERENCES Speakers(speaker_id)
);
CREATE TABLE Conferences (
    conference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date DATE
);
INSERT INTO "Conferences" VALUES(1,'Education Tech Summit','2023-01-01');
INSERT INTO "Conferences" VALUES(2,'Teaching Innovations Conference','2023-01-15');
INSERT INTO "Conferences" VALUES(3,'Learning & Development Expo','2023-02-01');
INSERT INTO "Conferences" VALUES(4,'eLearning World Forum','2023-02-15');
INSERT INTO "Conferences" VALUES(5,'Academic Research Symposium','2023-03-01');
CREATE TABLE "Sessions" (
    session_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_type TEXT,
    duration INTEGER 
);
INSERT INTO "Sessions" VALUES(1,'Keynote_Speech',60);
INSERT INTO "Sessions" VALUES(2,'Panel Discussion',90);
INSERT INTO "Sessions" VALUES(3,'Workshop',120);
INSERT INTO "Sessions" VALUES(4,'Breakout Session',60);
INSERT INTO "Sessions" VALUES(5,'Presentation',30);
CREATE TABLE Speakers (
    speaker_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    title TEXT
);
INSERT INTO "Speakers" VALUES(1,'John','Doe','CEO at Tech Inc.');
INSERT INTO "Speakers" VALUES(2,'Jane','Smith','Professor at University XYZ');
INSERT INTO "Speakers" VALUES(3,'Bob','Williams','Innovator at ABC Company');
INSERT INTO "Speakers" VALUES(4,'Alice','Johnson','Director at Learning Center PQR');
INSERT INTO "Speakers" VALUES(5,'Charlie','Brown','Researcher at Data Analytics Institute');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Conferences',5);
INSERT INTO "sqlite_sequence" VALUES('Speakers',5);
INSERT INTO "sqlite_sequence" VALUES('Conference_Sessions',5);
INSERT INTO "sqlite_sequence" VALUES('Sessions',5);
COMMIT;
