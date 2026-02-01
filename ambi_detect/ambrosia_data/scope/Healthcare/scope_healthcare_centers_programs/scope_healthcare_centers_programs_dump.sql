BEGIN TRANSACTION;
CREATE TABLE Appointments(
    AppointmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    PatientID INTEGER,
    DoctorID INTEGER,
    Date DATETIME,
    FOREIGN KEY(PatientID) REFERENCES Patients(PatientID),
    FOREIGN KEY(DoctorID) REFERENCES Doctors(DoctorID));
INSERT INTO "Appointments" VALUES(1,1,1,'2022-07-15 10:00:00');
INSERT INTO "Appointments" VALUES(2,2,2,'2022-07-18 14:00:00');
INSERT INTO "Appointments" VALUES(3,3,3,'2022-07-20 09:00:00');
INSERT INTO "Appointments" VALUES(4,4,4,'2022-07-22 15:00:00');
INSERT INTO "Appointments" VALUES(5,5,5,'2022-07-24 16:00:00');
CREATE TABLE Doctors(
    DoctorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Specialization TEXT,
    CenterID INTEGER,
    FOREIGN KEY(CenterID) REFERENCES Healthcare_Centers(CenterID));
INSERT INTO "Doctors" VALUES(1,'Dr. Smith','Cardiology',1);
INSERT INTO "Doctors" VALUES(2,'Dr. Johnson','Neurology',2);
INSERT INTO "Doctors" VALUES(3,'Dr. Williams','Orthopedics',3);
INSERT INTO "Doctors" VALUES(4,'Dr. Thompson','Pediatrics',4);
INSERT INTO "Doctors" VALUES(5,'Dr. Brown','Gastroenterology',5);
CREATE TABLE Healthcare_Centers(
    CenterID INTEGER PRIMARY KEY AUTOINCREMENT,
    CenterName TEXT,
    Address TEXT);
INSERT INTO "Healthcare_Centers" VALUES(1,'Center1','Address1');
INSERT INTO "Healthcare_Centers" VALUES(2,'Center2','Address2');
INSERT INTO "Healthcare_Centers" VALUES(3,'Center3','Address3');
INSERT INTO "Healthcare_Centers" VALUES(4,'Center4','Address4');
INSERT INTO "Healthcare_Centers" VALUES(5,'Center5','Address5');
CREATE TABLE Healthcare_Centers_Programs(
    CenterID INTEGER,
    ProgramID INTEGER,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(CenterID) REFERENCES Healthcare_Centers(CenterID),
    FOREIGN KEY(ProgramID) REFERENCES Programs(ProgramID));
INSERT INTO "Healthcare_Centers_Programs" VALUES(1,1,'2022-06-01','2022-08-31');
INSERT INTO "Healthcare_Centers_Programs" VALUES(1,2,'2022-09-01','2022-11-30');
INSERT INTO "Healthcare_Centers_Programs" VALUES(2,1,'2022-06-01','2022-08-31');
INSERT INTO "Healthcare_Centers_Programs" VALUES(2,2,'2022-09-01','2022-11-30');
INSERT INTO "Healthcare_Centers_Programs" VALUES(3,1,'2022-06-01','2022-08-31');
INSERT INTO "Healthcare_Centers_Programs" VALUES(3,2,'2022-09-01','2022-11-30');
INSERT INTO "Healthcare_Centers_Programs" VALUES(4,1,'2022-06-01','2022-08-31');
INSERT INTO "Healthcare_Centers_Programs" VALUES(4,2,'2022-09-01','2022-11-30');
INSERT INTO "Healthcare_Centers_Programs" VALUES(5,1,'2022-06-01','2022-08-31');
INSERT INTO "Healthcare_Centers_Programs" VALUES(5,2,'2022-09-01','2022-11-30');
CREATE TABLE MedicalHistory(
    HistoryID INTEGER PRIMARY KEY AUTOINCREMENT,
    PatientID INTEGER,
    Diagnosis TEXT,
    Treatment TEXT,
    FOREIGN KEY(PatientID) REFERENCES Patients(PatientID));
INSERT INTO "MedicalHistory" VALUES(1,1,'Diabetes','Insulin Injections');
INSERT INTO "MedicalHistory" VALUES(2,2,'High Blood Pressure','Medication');
INSERT INTO "MedicalHistory" VALUES(3,3,'Asthma','Inhaler');
INSERT INTO "MedicalHistory" VALUES(4,4,'Broken Arm','Cast');
INSERT INTO "MedicalHistory" VALUES(5,5,'Anemia','Iron Supplements');
CREATE TABLE Patients(
    PatientID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Age INTEGER,
    CenterID INTEGER,
    FOREIGN KEY(CenterID) REFERENCES Healthcare_Centers(CenterID));
INSERT INTO "Patients" VALUES(1,'John','Doe',30,1);
INSERT INTO "Patients" VALUES(2,'Jane','Smith',28,2);
INSERT INTO "Patients" VALUES(3,'Mike','Brown',26,3);
INSERT INTO "Patients" VALUES(4,'Emma','Wilson',32,4);
INSERT INTO "Patients" VALUES(5,'David','Jackson',40,5);
CREATE TABLE Programs(
    ProgramID INTEGER PRIMARY KEY AUTOINCREMENT,
    ProgramType TEXT);
INSERT INTO "Programs" VALUES(1,'Wellness_Initiatives');
INSERT INTO "Programs" VALUES(2,'Disease_Prevention');
INSERT INTO "Programs" VALUES(3,'Hospitality');
INSERT INTO "Programs" VALUES(4,'Fitness');
INSERT INTO "Programs" VALUES(5,'Emergency');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Healthcare_Centers',5);
INSERT INTO "sqlite_sequence" VALUES('Programs',5);
INSERT INTO "sqlite_sequence" VALUES('Patients',5);
INSERT INTO "sqlite_sequence" VALUES('Doctors',5);
INSERT INTO "sqlite_sequence" VALUES('Appointments',5);
INSERT INTO "sqlite_sequence" VALUES('MedicalHistory',5);
COMMIT;
