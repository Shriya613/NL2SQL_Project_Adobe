BEGIN TRANSACTION;
CREATE TABLE Appointments (
    AppointmentID INTEGER PRIMARY KEY,
    PatientID INTEGER,
    DoctorID INTEGER,
    Date DATE,
    Time TIME,
    FOREIGN KEY(PatientID) REFERENCES Patients(PatientID),
    FOREIGN KEY(DoctorID) REFERENCES Doctors(DoctorID));
INSERT INTO "Appointments" VALUES(1,1,1,'2022-07-10','10:00 AM');
INSERT INTO "Appointments" VALUES(2,2,2,'2022-07-12','03:00 PM');
INSERT INTO "Appointments" VALUES(3,3,3,'2022-07-14','11:00 AM');
CREATE TABLE Doctors (
    DoctorID INTEGER PRIMARY KEY,
    Name TEXT,
    Specialty TEXT,
    Founding_Year INT,
    Past_Owners TEXT,
    Healthcare_Facility TEXT);
INSERT INTO "Doctors" VALUES(1,'Dr. John Smith','Cardiology',2005,'John Brown, Mary Johnson','St. Luke''s Hospital');
INSERT INTO "Doctors" VALUES(2,'Dr. Jane Doe','Pediatrics',2010,'James White, Michael Black','Children''s Hospital');
INSERT INTO "Doctors" VALUES(3,'Dr. Susan Lee','Dermatology',2015,'Emma Green, David Wilson','Sunshine Skin Clinic');
CREATE TABLE Insurances (
    InsuranceID INTEGER PRIMARY KEY,
    PolicyNumber TEXT,
    Provider TEXT,
    CoverageAmount DECIMAL,
    StartDate DATE,
    EndDate DATE,
    PatientID INTEGER,
    FOREIGN KEY(PatientID) REFERENCES Patients(PatientID));
INSERT INTO "Insurances" VALUES(1,'ABCD1234','HealthPlus',10000,'2022-01-01','2023-12-31',1);
INSERT INTO "Insurances" VALUES(2,'EFGH5678','Wellness Care',12000,'2022-01-15','2023-12-14',2);
INSERT INTO "Insurances" VALUES(3,'IJKL9012','Family Plan',15000,'2022-02-01','2023-01-31',3);
CREATE TABLE Medications (
    MedicationID INTEGER PRIMARY KEY,
    MedicineName TEXT,
    DrugClass TEXT,
    Instructions TEXT);
INSERT INTO "Medications" VALUES(1,'Ibuprofen','Nonsteroidal Anti-Inflammatory Drugs (NSAIDs)','Take 200 mg every 4 to 6 hours as needed.');
INSERT INTO "Medications" VALUES(2,'Acetaminophen','Analgesics and Antipyretics','Take 500 mg every 4 to 6 hours as needed.');
CREATE TABLE Patients (
    PatientID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE);
INSERT INTO "Patients" VALUES(1,'Alice','Smith','1987-06-15');
INSERT INTO "Patients" VALUES(2,'Bob','Johnson','1990-02-20');
INSERT INTO "Patients" VALUES(3,'Carol','Williams','1985-10-05');
CREATE TABLE Prescriptions (
    PrescriptionID INTEGER PRIMARY KEY,
    PatientID INTEGER,
    MedicationID INTEGER,
    IssueDate DATE,
    ExpiryDate DATE,
    Quantity INT,
    FOREIGN KEY(PatientID) REFERENCES Patients(PatientID),
    FOREIGN KEY(MedicationID) REFERENCES Medications(MedicationID));
INSERT INTO "Prescriptions" VALUES(1,1,1,'2022-06-20','2022-06-30',30);
INSERT INTO "Prescriptions" VALUES(2,2,1,'2022-06-25','2022-07-09',60);
INSERT INTO "Prescriptions" VALUES(3,3,2,'2022-07-05','2022-07-19',14);
COMMIT;
