BEGIN TRANSACTION;
CREATE TABLE Appointments (
    AppointmentID INTEGER PRIMARY KEY,
    PatientID INTEGER,
    DoctorID INTEGER,
    AppointmentDate TIMESTAMP,
    ReasonForVisit TEXT,
    FOREIGN KEY(PatientID) REFERENCES Patients(PatientID),
    FOREIGN KEY(DoctorID) REFERENCES Doctors(DoctorID)
);
INSERT INTO "Appointments" VALUES(1,1,1,'2022-01-15 10:00:00','Routine Checkup');
INSERT INTO "Appointments" VALUES(2,2,2,'2022-02-10 14:00:00','Headache');
CREATE TABLE Doctors (
    DoctorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Specialization TEXT,
    PhoneNumber CHAR(10),
    EmailAddress TEXT
);
INSERT INTO "Doctors" VALUES(1,'Dr.','Jones','Cardiology','555-1234','drjones@example.com');
INSERT INTO "Doctors" VALUES(2,'Dr.','Williams','Neurology','555-5678','dwilliams@example.com');
CREATE TABLE Patients (
    PatientID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    InsuranceProvider TEXT,
    Address TEXT
);
INSERT INTO "Patients" VALUES(1,'John','Doe','1985-07-30','Blue Cross Blue Shield','123 Main St');
INSERT INTO "Patients" VALUES(2,'Jane','Smith','1990-06-15','Aetna','456 Elm St');
CREATE TABLE Prescriptions (
    PrescriptionID INTEGER PRIMARY KEY,
    AppointmentID INTEGER,
    MedicationName TEXT,
    Dosage NVARCHAR(20),
    Instructions TEXT,
    FOREIGN KEY(AppointmentID) REFERENCES Appointments(AppointmentID)
);
INSERT INTO "Prescriptions" VALUES(1,1,'Ibuprofen','2 tablets every 6 hours','Take with food');
INSERT INTO "Prescriptions" VALUES(2,2,'Amoxicillin','500mg twice a day','Take with water on an empty stomach');
CREATE TABLE "SurgicalProcedures" (
    ProcedureID INTEGER PRIMARY KEY,
    Description TEXT,
    "Cost" NUMERIC,
    TimeRequired TEXT
);
INSERT INTO "SurgicalProcedures" VALUES(1,'Knee Arthroscopy',5000,'2 hours');
INSERT INTO "SurgicalProcedures" VALUES(2,'Appendectomy',4000,'1 hour');
COMMIT;
