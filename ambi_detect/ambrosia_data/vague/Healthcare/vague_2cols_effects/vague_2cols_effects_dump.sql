BEGIN TRANSACTION;
CREATE TABLE Appointments (
    AppointmentID INTEGER PRIMARY KEY,
    PatientID INTEGER REFERENCES Patients(PatientID),
    DoctorID INTEGER REFERENCES Doctors(DoctorID),
    AppointmentDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ReasonForVisit TEXT,
    IsCancelled BOOLEAN DEFAULT FALSE);
INSERT INTO "Appointments" VALUES(1,1,1,'2024-02-19 12:14:44','Routine checkup',0);
CREATE TABLE Diagnoses (
    DiagnosisID INTEGER PRIMARY KEY,
    PatientID INTEGER REFERENCES Patients(PatientID),
    DoctorID INTEGER REFERENCES Doctors(DoctorID),
    Code TEXT,
    Description TEXT);
INSERT INTO "Diagnoses" VALUES(1,1,1,'A123','Hypertension');
CREATE TABLE Doctors (
    DoctorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Specialty TEXT,
    LicenseNumber VARCHAR(20),
    OfficePhone TEXT UNIQUE);
INSERT INTO "Doctors" VALUES(1,'Jane','Smith','Cardiology','LIC001234567','+0987654321');
CREATE TABLE Medications (
    DrugID INTEGER PRIMARY KEY,
    Name TEXT,
    Classification TEXT,
    Manufacturer TEXT,
    Side_Effects TEXT,
    Effectiveness TEXT);
INSERT INTO "Medications" VALUES(1,'Aspirin','Analgesic','Bayer','Gastrointestinal upset','High');
CREATE TABLE Patients (
    PatientID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DOB Date ,
    PhoneNumber VARCHAR(15),
    EmailAddress VARCHAR(255) UNIQUE);
INSERT INTO "Patients" VALUES(1,'John','Doe','1987-06-15','+1234567890','john.doe@example.com');
CREATE TABLE Prescriptions (
    PrescriptionID INTEGER PRIMARY KEY,
    PatientID INTEGER REFERENCES Patients(PatientID),
    DoctorID INTEGER REFERENCES Doctors(DoctorID),
    MedicationID INTEGER REFERENCES Medications(MedicationID),
    StartDate Date ,
    EndDate Date,
    Quantity INTEGER );
INSERT INTO "Prescriptions" VALUES(1,1,1,1,'2022-01-01','2022-01-31',10);
COMMIT;
