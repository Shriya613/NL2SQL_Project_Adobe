BEGIN TRANSACTION;
CREATE TABLE "Doctors"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    specialist_doctor TEXT,
    hospital_id INTEGER REFERENCES Hospitals(id));
INSERT INTO "Doctors" VALUES(1,'Dr. Jane Smith','Radiologist',1);
INSERT INTO "Doctors" VALUES(2,'Dr. John Brown','Radiologist',2);
INSERT INTO "Doctors" VALUES(3,'Ms. Mary Johnson','Ultrasound Technician',1);
INSERT INTO "Doctors" VALUES(4,'Mr. Mike Davis','Ultrasound Technician',2);
INSERT INTO "Doctors" VALUES(5,'Dr. Sarah Wilson','Cardiologist',3);
INSERT INTO "Doctors" VALUES(6,'Dr. James White','Gastroenterologist',4);
INSERT INTO "Doctors" VALUES(7,'Dr. Jennifer Green','Neurologist',1);
INSERT INTO "Doctors" VALUES(8,'Dr. David Black','Orthopedic Surgeon',2);
CREATE TABLE Hospitals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Hospitals" VALUES(1,'Western Medical Center');
INSERT INTO "Hospitals" VALUES(2,'Eastern General Hospital');
INSERT INTO "Hospitals" VALUES(3,'Northern Healthcare');
INSERT INTO "Hospitals" VALUES(4,'Central Care Clinic');
INSERT INTO "Hospitals" VALUES(5,'City Urgent Care');
CREATE TABLE Jobs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    location TEXT,
    salary INTEGER,
    hospital_id INTEGER REFERENCES Hospitals(id));
INSERT INTO "Jobs" VALUES(1,'CT Scan Technician','Perform CT scans on patients.','New York City',90000,1);
INSERT INTO "Jobs" VALUES(2,'MRI Tech','Operate MRI machines.','New York City',85000,1);
INSERT INTO "Jobs" VALUES(3,'ER Physician','Treat emergency cases.','New York City',200000,1);
INSERT INTO "Jobs" VALUES(4,'Anesthesia Nurse','Monitor patient during surgeries.','New York City',100000,1);
INSERT INTO "Jobs" VALUES(5,'Pharmacist','Dispense medications.','New York City',95000,1);
INSERT INTO "Jobs" VALUES(6,'Family Medicine Doctor','Diagnose and treat illnesses.','Chicago',160000,2);
INSERT INTO "Jobs" VALUES(7,'Dietitian','Create diet plans for patients.','Chicago',70000,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hospitals',5);
INSERT INTO "sqlite_sequence" VALUES('Jobs',7);
INSERT INTO "sqlite_sequence" VALUES('Doctors',8);
COMMIT;
