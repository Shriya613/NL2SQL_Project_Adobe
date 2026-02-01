BEGIN TRANSACTION;
CREATE TABLE Agency(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Agency" VALUES(1,'Expedia');
INSERT INTO "Agency" VALUES(2,'Booking.com');
CREATE TABLE Applications(
    tourGuideId INTEGER,
    travelAgentId INTEGER,
    jobId INTEGER,
    applicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(tourGuideId) REFERENCES Tour_Guide(id),
    FOREIGN KEY(travelAgentId) REFERENCES Travel_Agent(id),
    FOREIGN KEY(jobId) REFERENCES Jobs(id));
INSERT INTO "Applications" VALUES(1,3,1,'2024-03-13 17:41:57');
INSERT INTO "Applications" VALUES(2,4,2,'2024-03-13 17:41:57');
INSERT INTO "Applications" VALUES(3,1,3,'2024-03-13 17:41:57');
INSERT INTO "Applications" VALUES(4,2,4,'2024-03-13 17:41:57');
INSERT INTO "Applications" VALUES(5,1,5,'2024-03-13 17:41:57');
CREATE TABLE Jobs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT);
INSERT INTO "Jobs" VALUES(1,'Software Engineer','Developing web applications.');
INSERT INTO "Jobs" VALUES(2,'Data Analyst','Analyzing customer behavior.');
INSERT INTO "Jobs" VALUES(3,'Graphic Designer','Creating visuals for marketing materials.');
INSERT INTO "Jobs" VALUES(4,'Marketing Specialist','Promoting products and services.');
INSERT INTO "Jobs" VALUES(5,'HR Manager','Managing employees and recruitment.');
CREATE TABLE Skills(
    skillId INTEGER PRIMARY KEY AUTOINCREMENT,
    skillName TEXT UNIQUE);
INSERT INTO "Skills" VALUES(1,'Python Programming');
INSERT INTO "Skills" VALUES(2,'SQL Database Management');
INSERT INTO "Skills" VALUES(3,'JavaScript Development');
INSERT INTO "Skills" VALUES(4,'SEO Optimization');
INSERT INTO "Skills" VALUES(5,'Customer Relationship Management');
CREATE TABLE Tour_Guide(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guideName TEXT,
    agencyId INTEGER,
    FOREIGN KEY(agencyId) REFERENCES Agency(id));
INSERT INTO "Tour_Guide" VALUES(1,'John Doe',1);
INSERT INTO "Tour_Guide" VALUES(2,'Jane Smith',2);
INSERT INTO "Tour_Guide" VALUES(3,'Michael Johnson',1);
INSERT INTO "Tour_Guide" VALUES(4,'Emily Davis',2);
INSERT INTO "Tour_Guide" VALUES(5,'David Brown',1);
CREATE TABLE Travel_Agent(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agentName TEXT,
    agencyId INTEGER,
    FOREIGN KEY(agencyId) REFERENCES Agency(id));
INSERT INTO "Travel_Agent" VALUES(1,'Alice Lee',1);
INSERT INTO "Travel_Agent" VALUES(2,'Bob White',2);
INSERT INTO "Travel_Agent" VALUES(3,'Carol Black',1);
INSERT INTO "Travel_Agent" VALUES(4,'Daniel Green',2);
INSERT INTO "Travel_Agent" VALUES(5,'Eva Blue',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Agency',2);
INSERT INTO "sqlite_sequence" VALUES('Tour_Guide',5);
INSERT INTO "sqlite_sequence" VALUES('Travel_Agent',5);
INSERT INTO "sqlite_sequence" VALUES('Jobs',5);
INSERT INTO "sqlite_sequence" VALUES('Skills',5);
COMMIT;
