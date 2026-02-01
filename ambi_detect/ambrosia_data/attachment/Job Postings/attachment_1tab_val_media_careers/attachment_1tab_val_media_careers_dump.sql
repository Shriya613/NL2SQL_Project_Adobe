BEGIN TRANSACTION;
CREATE TABLE Company (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    phoneNumber TEXT,
    website TEXT);
INSERT INTO "Company" VALUES(1,'Newspaper Co.','123 Main St','555-0198','https://www.newspapercopublishers.com');
INSERT INTO "Company" VALUES(2,'Broadcast News Inc.','456 Elm St','555-0197','https://www.broadcastnewsinc.net');
INSERT INTO "Company" VALUES(3,'Digital Media Hub','789 Oak St','555-0196','https://www.digitalmediahubpublishers.org');
INSERT INTO "Company" VALUES(4,'Print Journalists Network','101 Pine St','555-0195','https://www.printjournalistsnetwork.info');
INSERT INTO "Company" VALUES(5,'Newsroom Professionals Association','202 Fir St','555-0194','https://www.newsroomprofessionalsassociation.biz');
CREATE TABLE JobPosting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    mediaCareer TEXT,
    languages TEXT,
    startDate DATE,
    endDate DATE,
    companyId INTEGER,
    locationId INTEGER,
    FOREIGN KEY (companyId) REFERENCES Company(id),
    FOREIGN KEY (locationId) REFERENCES Location(id));
INSERT INTO "JobPosting" VALUES(1,'Senior Journalist','Experienced journalist required','Journalist','English and Spanish','2022-01-01','2022-12-31',1,1);
INSERT INTO "JobPosting" VALUES(2,'Freelance Journalist','Contributor needed for part-time work','Journalist','Italian','2022-01-01','2022-12-31',2,2);
INSERT INTO "JobPosting" VALUES(3,'Chief Editor','Looking for a seasoned editor','Editor','English and Spanish','2022-01-01','2022-12-31',3,3);
INSERT INTO "JobPosting" VALUES(4,'Sub-editor','Assistant editor role available','Editor','German','2022-01-01','2022-12-31',4,4);
INSERT INTO "JobPosting" VALUES(5,'Multimedia Designer','Design skills required','Graphic Designer','Portuguese','2022-01-01','2022-12-31',5,5);
INSERT INTO "JobPosting" VALUES(6,'Content Writer','Need someone to write news articles','Writer','None specified','2022-01-01','2022-12-31',1,1);
INSERT INTO "JobPosting" VALUES(7,'Photographer','Capturing images for print/online publications','Photographer','French','2022-01-01','2022-12-31',2,2);
INSERT INTO "JobPosting" VALUES(8,'Video Producer','Produce video content for online platforms','Videographer','Russian','2022-01-01','2022-12-31',3,3);
INSERT INTO "JobPosting" VALUES(9,'Web Developer','Developing websites for various clients','Web Developer','None specified','2022-01-01','2022-12-31',4,4);
INSERT INTO "JobPosting" VALUES(10,'Data Analyst','Analyze data for marketing purposes','Analyst','Spanish','2022-01-01','2022-12-31',5,5);
CREATE TABLE Location (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    state TEXT,
    country TEXT,
    zipCode TEXT);
INSERT INTO "Location" VALUES(1,'New York','NY','USA','10001');
INSERT INTO "Location" VALUES(2,'Los Angeles','CA','USA','90001');
INSERT INTO "Location" VALUES(3,'London','England','UK','W1A 1AA');
INSERT INTO "Location" VALUES(4,'Paris','Île-de-France','France','75001');
INSERT INTO "Location" VALUES(5,'Tokyo','Japan','JP','100-0001');
CREATE TABLE Skills(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skillName TEXT UNIQUE);
INSERT INTO "Skills" VALUES(1,'Communication');
INSERT INTO "Skills" VALUES(2,'Writing');
INSERT INTO "Skills" VALUES(3,'Editing');
INSERT INTO "Skills" VALUES(4,'Research');
INSERT INTO "Skills" VALUES(5,'Marketing');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Company',5);
INSERT INTO "sqlite_sequence" VALUES('Location',5);
INSERT INTO "sqlite_sequence" VALUES('Skills',5);
INSERT INTO "sqlite_sequence" VALUES('JobPosting',10);
COMMIT;
