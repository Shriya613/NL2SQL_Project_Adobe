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
CREATE TABLE "Drug" (
    drugID INTEGER PRIMARY KEY,
    name TEXT,
    dosage TEXT,
    sideEffectID INTEGER,
    FOREIGN KEY(sideEffectID) REFERENCES "Effects"(sideEffectID));
INSERT INTO "Drug" VALUES(1,'Lisinopril','5mg',1);
INSERT INTO "Drug" VALUES(2,'Sumatriptan','50mg',2);
INSERT INTO "Drug" VALUES(3,'Aspirin','325mg',3);
CREATE TABLE "Effects" (
    sideEffectID INTEGER PRIMARY KEY,
    effectText TEXT,
    severity TEXT 
);
INSERT INTO "Effects" VALUES(1,'Nausea','Moderate');
INSERT INTO "Effects" VALUES(2,'Headache','Mild');
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
COMMIT;
