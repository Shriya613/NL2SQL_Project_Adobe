BEGIN TRANSACTION;
CREATE TABLE Applications(
    id INTEGER PRIMARY KEY,
    applicantId INTEGER,
    postingId INTEGER,
    applicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (applicantId) REFERENCES Companies(id),
    FOREIGN KEY (postingId) REFERENCES Postings(id));
INSERT INTO "Applications" VALUES(1,1,1,'2024-03-13 17:03:39');
INSERT INTO "Applications" VALUES(2,2,2,'2024-03-13 17:03:39');
CREATE TABLE Call_Center_Agents(
    id INTEGER PRIMARY KEY,
    skills TEXT, FullName TEXT);
INSERT INTO "Call_Center_Agents" VALUES(1,'Proactivity','Mike Brown');
INSERT INTO "Call_Center_Agents" VALUES(2,'Multitasking','Emily White');
CREATE TABLE Companies(
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT);
INSERT INTO "Companies" VALUES(1,'XYZ Inc.','New York');
INSERT INTO "Companies" VALUES(2,'ABC Corporation','Los Angeles');
CREATE TABLE Customer_Service_Representatives(
    id INTEGER PRIMARY KEY,
    skills TEXT, FullName TEXT);
INSERT INTO "Customer_Service_Representatives" VALUES(1,'Proactivity','John Doe');
INSERT INTO "Customer_Service_Representatives" VALUES(2,'Communication Skills','Jane Smith');
CREATE TABLE Postings(
    id INTEGER PRIMARY KEY,
    title TEXT,
    companyID INTEGER,
    FOREIGN KEY (companyID) REFERENCES Companies(id));
INSERT INTO "Postings" VALUES(1,'CSR',1);
INSERT INTO "Postings" VALUES(2,'Call Center Agent',2);
COMMIT;
