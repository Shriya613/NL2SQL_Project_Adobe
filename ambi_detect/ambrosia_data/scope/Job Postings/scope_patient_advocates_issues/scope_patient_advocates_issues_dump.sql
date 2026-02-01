BEGIN TRANSACTION;
CREATE TABLE Issues(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_name TEXT,
    description TEXT);
INSERT INTO "Issues" VALUES(1,'Billing','Handling insurance claims and billing issues.');
INSERT INTO "Issues" VALUES(2,'Medical Records','Managing and maintaining patients medical records.');
INSERT INTO "Issues" VALUES(3,'Appointment Scheduling','Coordinating appointments for patients and healthcare providers.');
INSERT INTO "Issues" VALUES(4,'Emergency Assistance','Assisting patients during emergencies and coordinating care.');
INSERT INTO "Issues" VALUES(5,'Health Insurance','Helping patients navigate their health insurance options.');
CREATE TABLE Patient_Advocates(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Patient_Advocates" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Patient_Advocates" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Patient_Advocates" VALUES(3,'Michael Johnson','michaeljohnson@example.com');
INSERT INTO "Patient_Advocates" VALUES(4,'Emily Davis','emilydavis@example.com');
INSERT INTO "Patient_Advocates" VALUES(5,'David Brown','davidbrown@example.com');
CREATE TABLE Patient_Advocates_Issues(
    patient_advocate_id INTEGER REFERENCES Patient_Advocates(id) ON DELETE CASCADE,
    issue_id INTEGER REFERENCES Issues(id));
INSERT INTO "Patient_Advocates_Issues" VALUES(1,1);
INSERT INTO "Patient_Advocates_Issues" VALUES(2,1);
INSERT INTO "Patient_Advocates_Issues" VALUES(3,1);
INSERT INTO "Patient_Advocates_Issues" VALUES(4,1);
INSERT INTO "Patient_Advocates_Issues" VALUES(5,1);
INSERT INTO "Patient_Advocates_Issues" VALUES(5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Patient_Advocates',5);
INSERT INTO "sqlite_sequence" VALUES('Issues',5);
COMMIT;
