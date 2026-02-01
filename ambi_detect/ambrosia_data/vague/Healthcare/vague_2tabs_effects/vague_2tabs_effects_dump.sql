BEGIN TRANSACTION;
CREATE TABLE Doctor (
    doctorID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    specialty TEXT
);
INSERT INTO "Doctor" VALUES(1,'Dr.','Ahn','Cardiology');
INSERT INTO "Doctor" VALUES(2,'Dr.','Martinez','Neurology');
INSERT INTO "Doctor" VALUES(3,'Dr.','Lee','Pediatrics');
CREATE TABLE Drug (
    drugID INTEGER PRIMARY KEY,
    name TEXT,
    dosage TEXT,
    sideEffectID INTEGER,
    effectivenessID INTEGER,
    FOREIGN KEY(sideEffectID) REFERENCES Side_Effects(sideEffectID),
    FOREIGN KEY(effectivenessID) REFERENCES Effectiveness(effectivenessID)
);
INSERT INTO "Drug" VALUES(1,'Lisinopril','5mg',1,1);
INSERT INTO "Drug" VALUES(2,'Sumatriptan','50mg',2,2);
INSERT INTO "Drug" VALUES(3,'Aspirin','325mg',3,3);
CREATE TABLE Effectiveness (
    effectivenessID INTEGER PRIMARY KEY,
    percentage INTEGER 
);
INSERT INTO "Effectiveness" VALUES(1,80);
INSERT INTO "Effectiveness" VALUES(2,70);
INSERT INTO "Effectiveness" VALUES(3,90);
CREATE TABLE MedicalRecord (
    recordID INTEGER PRIMARY KEY,
    patientID INTEGER,
    diagnosis TEXT,
    treatmentPlan TEXT,
    FOREIGN KEY(patientID) REFERENCES Patient(patientID)
);
INSERT INTO "MedicalRecord" VALUES(1,1,'Hypertension','Monitor blood pressure and take medication as prescribed.');
INSERT INTO "MedicalRecord" VALUES(2,2,'Migraine','Take pain relievers and avoid known triggers.');
CREATE TABLE Patient (
    patientID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE
);
INSERT INTO "Patient" VALUES(1,'John','Doe','2000-07-16');
INSERT INTO "Patient" VALUES(2,'Jane','Smith','1985-03-20');
INSERT INTO "Patient" VALUES(3,'Emily','Johnson','1998-01-12');
CREATE TABLE Prescription (
    prescriptionID INTEGER PRIMARY KEY,
    patientID INTEGER,
    drugID INTEGER,
    quantity INTEGER,
    FOREIGN KEY(patientID) REFERENCES Patient(patientID),
    FOREIGN KEY(drugID) REFERENCES Drug(drugID)
);
INSERT INTO "Prescription" VALUES(1,1,1,30);
INSERT INTO "Prescription" VALUES(2,2,2,15);
INSERT INTO "Prescription" VALUES(3,3,3,45);
CREATE TABLE Side_Effects (
    sideEffectID INTEGER PRIMARY KEY,
    effectText TEXT,
    severity TEXT 
);
INSERT INTO "Side_Effects" VALUES(1,'Nausea','Moderate');
INSERT INTO "Side_Effects" VALUES(2,'Headache','Mild');
COMMIT;
