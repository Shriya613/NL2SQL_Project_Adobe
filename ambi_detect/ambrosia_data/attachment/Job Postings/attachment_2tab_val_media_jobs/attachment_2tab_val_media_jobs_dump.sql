BEGIN TRANSACTION;
CREATE TABLE Applications(
    applicationId INTEGER PRIMARY KEY AUTOINCREMENT,
    applicantEmail TEXT,
    appliedJob TEXT,
    applyDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
INSERT INTO "Applications" VALUES(1,'john@example.com','Reporter','2023-04-17 12:30:00');
INSERT INTO "Applications" VALUES(2,'jane@example.com','Copy Editor','2023-04-17 12:35:00');
INSERT INTO "Applications" VALUES(3,'alice@example.com','Web Producer','2023-04-17 12:40:00');
INSERT INTO "Applications" VALUES(4,'bob@example.com','Investigative Reporter','2023-04-17 12:45:00');
INSERT INTO "Applications" VALUES(5,'carol@example.com','Photographer','2023-04-17 12:50:00');
CREATE TABLE Companies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    companyName TEXT,
    location TEXT);
INSERT INTO "Companies" VALUES(1,'The New York Times','New York');
INSERT INTO "Companies" VALUES(2,'The Guardian','London');
INSERT INTO "Companies" VALUES(3,'Le Monde','Paris');
INSERT INTO "Companies" VALUES(4,'Der Spiegel','Hamburg');
INSERT INTO "Companies" VALUES(5,'La Repubblica','Milan');
CREATE TABLE Editors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    language_proficiency TEXT ,
    experience_years INT);
INSERT INTO "Editors" VALUES(1,'Michael Jackson','michael@example.com','English',10);
INSERT INTO "Editors" VALUES(2,'Karen Clark','karen@example.com','Chinese',9);
INSERT INTO "Editors" VALUES(3,'David Lee','david@example.com','Arabic',5);
INSERT INTO "Editors" VALUES(4,'Elizabeth Green','elizabeth@example.com','Russian',8);
INSERT INTO "Editors" VALUES(5,'Frank Miller','frank@example.com','Portuguese',6);
CREATE TABLE JobPostings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jobTitle TEXT,
    description TEXT,
    salary DECIMAL(10,2) );
INSERT INTO "JobPostings" VALUES(1,'Reporter','Report on breaking news stories.',50000);
INSERT INTO "JobPostings" VALUES(2,'Copy Editor','Check articles for grammar and factual errors.',55000);
INSERT INTO "JobPostings" VALUES(3,'Web Producer','Manage website content and social media.',60000);
INSERT INTO "JobPostings" VALUES(4,'Investigative Reporter','Conduct in-depth investigations into important issues.',65000);
INSERT INTO "JobPostings" VALUES(5,'Photographer','Capture images for publication in print and online.',52000);
CREATE TABLE Journalists(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    language_proficiency TEXT ,
    years_experience INT);
INSERT INTO "Journalists" VALUES(1,'John Doe','john@example.com','English',3);
INSERT INTO "Journalists" VALUES(2,'Jane Smith','jane@example.com','French',6);
INSERT INTO "Journalists" VALUES(3,'Alice Johnson','alice@example.com','Spanish',8);
INSERT INTO "Journalists" VALUES(4,'Bob Brown','bob@example.com','German',7);
INSERT INTO "Journalists" VALUES(5,'Carol White','carol@example.com','Italian',4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Journalists',5);
INSERT INTO "sqlite_sequence" VALUES('Editors',5);
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',5);
INSERT INTO "sqlite_sequence" VALUES('Applications',5);
COMMIT;
