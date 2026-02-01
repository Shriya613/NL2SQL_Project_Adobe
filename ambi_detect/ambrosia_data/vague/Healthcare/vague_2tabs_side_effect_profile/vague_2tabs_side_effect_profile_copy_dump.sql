BEGIN TRANSACTION;
CREATE TABLE "Antibiotic" (
    antibioticID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    commonSideEffects INTEGER[],
    FOREIGN KEY(commonSideEffects) REFERENCES "Side_Effect_Profile"(sideEffectID));
INSERT INTO "Antibiotic" VALUES(1,'Amoxicillin',1);
INSERT INTO "Antibiotic" VALUES(2,'Azithromycin',5);
INSERT INTO "Antibiotic" VALUES(3,'Cephalexin',5);
INSERT INTO "Antibiotic" VALUES(4,'Clindamycin',2);
INSERT INTO "Antibiotic" VALUES(5,'Erythromycin',1);
CREATE TABLE Doctor (
    doctorID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Doctor" VALUES(1,'Dr.','Aaron');
INSERT INTO "Doctor" VALUES(2,'Dr.','Elizabeth');
INSERT INTO "Doctor" VALUES(3,'Dr.','Robert');
INSERT INTO "Doctor" VALUES(4,'Dr.','Nancy');
INSERT INTO "Doctor" VALUES(5,'Dr.','Kenneth');
CREATE TABLE Patient (
    patientID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Patient" VALUES(1,'John','Doe');
INSERT INTO "Patient" VALUES(2,'Jane','Smith');
INSERT INTO "Patient" VALUES(3,'Michael','Brown');
INSERT INTO "Patient" VALUES(4,'Emily','White');
INSERT INTO "Patient" VALUES(5,'David','Black');
CREATE TABLE Prescription (
    prescriptionID INTEGER PRIMARY KEY AUTOINCREMENT,
    antibioticID INTEGER,
    patientID INTEGER,
    doctorID INTEGER,
    datePrescribed DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(antibioticID) REFERENCES Antibiotic(antibioticID),
    FOREIGN KEY(patientID) REFERENCES Patient(patientID),
    FOREIGN KEY(doctorID) REFERENCES Doctor(doctorID));
INSERT INTO "Prescription" VALUES(1,1,1,1,'2024-02-19 12:44:17');
INSERT INTO "Prescription" VALUES(2,2,2,2,'2024-02-19 12:44:17');
INSERT INTO "Prescription" VALUES(3,3,3,3,'2024-02-19 12:44:17');
INSERT INTO "Prescription" VALUES(4,4,4,4,'2024-02-19 12:44:17');
INSERT INTO "Prescription" VALUES(5,5,5,5,'2024-02-19 12:44:17');
CREATE TABLE "Side_Effect_Profile" (
    sideEffectID INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT);
INSERT INTO "Side_Effect_Profile" VALUES(1,'Diarrhea');
INSERT INTO "Side_Effect_Profile" VALUES(2,'Vomiting');
INSERT INTO "Side_Effect_Profile" VALUES(3,'Nausea');
INSERT INTO "Side_Effect_Profile" VALUES(4,'Headache');
INSERT INTO "Side_Effect_Profile" VALUES(5,'Dizziness');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Patient',5);
INSERT INTO "sqlite_sequence" VALUES('Doctor',5);
INSERT INTO "sqlite_sequence" VALUES('Side_Effect_Profile',5);
INSERT INTO "sqlite_sequence" VALUES('Prescription',5);
INSERT INTO "sqlite_sequence" VALUES('Antibiotic',5);
COMMIT;
