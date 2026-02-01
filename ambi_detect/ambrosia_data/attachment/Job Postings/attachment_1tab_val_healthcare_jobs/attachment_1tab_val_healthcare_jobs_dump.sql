BEGIN TRANSACTION;
CREATE TABLE Employers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT);
INSERT INTO "Employers" VALUES(1,'Employer A','123 Main St');
INSERT INTO "Employers" VALUES(2,'Employer B','456 Elm St');
INSERT INTO "Employers" VALUES(3,'Employer C','789 Oak St');
CREATE TABLE JobPostings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    healthcare JOBS TEXT, -- assuming job titles can be stored as text
    shift Hours TEXT, -- assuming shifts can also be stored as text
    employerId INTEGER,
    locationId INTEGER,
    skillsIds TEXT, -- storing multiple skills per posting
    requirementsIds TEXT, -- storing multiple requirements per posting
    FOREIGN KEY (employerId) REFERENCES Employers(id),
    FOREIGN KEY (locationId) REFERENCES Locations(id));
INSERT INTO "JobPostings" VALUES(1,'Job 1','Description of Job 1','Nurse','Day Shift',1,1,'A,B','1,2');
INSERT INTO "JobPostings" VALUES(2,'Job 2','Description of Job 2','Nurse','Night Shift',1,2,'C,D','3,4');
INSERT INTO "JobPostings" VALUES(3,'Job 3','Description of Job 3','Physical Therapist','Day Shift',2,1,'E','5');
INSERT INTO "JobPostings" VALUES(4,'Job 4','Description of Job 4','Physical Therapist','Night Shift',2,2,'','6');
INSERT INTO "JobPostings" VALUES(5,'Job 5','Description of Job 5','Software Engineer','Flexible',3,3,'A,C,E','7,8');
CREATE TABLE Locations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    state TEXT);
INSERT INTO "Locations" VALUES(1,'City X','State AA');
INSERT INTO "Locations" VALUES(2,'City Y','State BB');
INSERT INTO "Locations" VALUES(3,'City Z','State CC');
CREATE TABLE Requirements(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requirement TEXT);
INSERT INTO "Requirements" VALUES(1,'Requirement 1');
INSERT INTO "Requirements" VALUES(2,'Requirement 2');
INSERT INTO "Requirements" VALUES(3,'Requirement 3');
INSERT INTO "Requirements" VALUES(4,'Requirement 4');
INSERT INTO "Requirements" VALUES(5,'Requirement 5');
CREATE TABLE Skills(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skillName TEXT);
INSERT INTO "Skills" VALUES(1,'Skill A');
INSERT INTO "Skills" VALUES(2,'Skill B');
INSERT INTO "Skills" VALUES(3,'Skill C');
INSERT INTO "Skills" VALUES(4,'Skill D');
INSERT INTO "Skills" VALUES(5,'Skill E');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Employers',3);
INSERT INTO "sqlite_sequence" VALUES('Locations',3);
INSERT INTO "sqlite_sequence" VALUES('Skills',5);
INSERT INTO "sqlite_sequence" VALUES('Requirements',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',5);
COMMIT;
