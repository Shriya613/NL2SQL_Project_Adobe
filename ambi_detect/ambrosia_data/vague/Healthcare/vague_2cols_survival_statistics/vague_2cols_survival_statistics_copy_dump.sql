BEGIN TRANSACTION;
CREATE TABLE Appointments (
    appointmentId INTEGER PRIMARY KEY AUTOINCREMENT,
    patientId INTEGER,
    doctorId INTEGER,
    appointmentDate TIMESTAMP,
    reasonForAppointment TEXT,
    outcome TEXT,
    FOREIGN KEY (patientId) REFERENCES Patients(id),
    FOREIGN KEY (doctorId) REFERENCES Doctors(doctorID));
INSERT INTO "Appointments" VALUES(1,1,1,'2022-02-01 10:00:00','Regular checkup','Healthy');
INSERT INTO "Appointments" VALUES(2,2,2,'2021-11-15 14:00:00','Flu symptoms','Recovering');
CREATE TABLE Diagnoses (
    diagnosisID INTEGER PRIMARY KEY AUTOINCREMENT,
    patientId INTEGER,
    illnessId INTEGER,
    diagnosedDate DATE,
    FOREIGN KEY (patientId) REFERENCES Patients(id),
    FOREIGN KEY (illnessId) REFERENCES Illnesses(illnessID));
INSERT INTO "Diagnoses" VALUES(1,1,1,'2022-01-15');
INSERT INTO "Diagnoses" VALUES(2,2,2,'2021-09-01');
CREATE TABLE Doctors (
    doctorID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    specialty TEXT,
    yearsExperience INT DEFAULT 0);
INSERT INTO "Doctors" VALUES(1,'Dr.','Jones','Cardiology',10);
INSERT INTO "Doctors" VALUES(2,'Dr.','Williams','Pediatrics',15);
CREATE TABLE "Illnesses" (
    illnessID INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    Survival_Statistics REAL,
    treatmentOption TEXT);
INSERT INTO "Illnesses" VALUES(1,'Breast Cancer','A disease where malignant cells form in the breast tissue.',85.0,'Chemotherapy and radiation therapy');
INSERT INTO "Illnesses" VALUES(2,'Diabetes','A chronic condition characterized by high blood sugar levels.',85.0,'Diet, exercise, and insulin therapy');
CREATE TABLE Patients (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE,
    gender CHAR(1) ,
    contactNumber VARCHAR(20),
    address TEXT);
INSERT INTO "Patients" VALUES(1,'John','Doe','1980-06-15','M','+1234567890','123 Main St');
INSERT INTO "Patients" VALUES(2,'Jane','Smith','1990-07-25','F','+0987654321','456 Elm St');
CREATE TABLE Prescriptions (
    prescriptionId INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosisId INTEGER,
    medication TEXT,
    dosage TEXT,
    frequency TEXT,
    startDate DATE,
    endDate DATE,
    FOREIGN KEY (diagnosisId) REFERENCES Diagnoses(diagnosisID));
INSERT INTO "Prescriptions" VALUES(1,1,'Tamoxifen','20mg','Once daily','2022-01-16','2022-12-31');
INSERT INTO "Prescriptions" VALUES(2,2,'Metformin','500mg','Twice daily','2021-09-02','2022-08-31');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Diagnoses',2);
INSERT INTO "sqlite_sequence" VALUES('Prescriptions',2);
INSERT INTO "sqlite_sequence" VALUES('Appointments',2);
COMMIT;
