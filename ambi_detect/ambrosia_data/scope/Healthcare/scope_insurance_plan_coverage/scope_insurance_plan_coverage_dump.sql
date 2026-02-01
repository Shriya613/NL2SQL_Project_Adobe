BEGIN TRANSACTION;
CREATE TABLE Appointment(AppointmentID INTEGER PRIMARY KEY AUTOINCREMENT, Date TIMESTAMP, Reason TEXT, PatientID INTEGER, DoctorID INTEGER, FOREIGN KEY(PatientID) REFERENCES Patient(PatientID), FOREIGN KEY(DoctorID) REFERENCES Doctor(DoctorID));
INSERT INTO "Appointment" VALUES(1,'2022-01-01 10:00:00','Annual checkup',1,1);
INSERT INTO "Appointment" VALUES(2,'2022-02-01 14:00:00','Follow-up visit',2,2);
INSERT INTO "Appointment" VALUES(3,'2022-03-01 09:00:00','Consultation',3,3);
INSERT INTO "Appointment" VALUES(4,'2022-04-01 11:00:00','Wound care',4,4);
INSERT INTO "Appointment" VALUES(5,'2022-05-01 13:00:00','Diabetes management',5,5);
CREATE TABLE Coverage(
    CoverageID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT);
INSERT INTO "Coverage" VALUES(1,'Prescription Drugs');
INSERT INTO "Coverage" VALUES(2,'Preventive Care');
INSERT INTO "Coverage" VALUES(3,'Hospitalization');
INSERT INTO "Coverage" VALUES(4,'Mental Health Services');
INSERT INTO "Coverage" VALUES(5,'Laboratory Tests');
CREATE TABLE Doctor(DoctorID INTEGER PRIMARY KEY AUTOINCREMENT, Name TEXT, Specialty TEXT);
INSERT INTO "Doctor" VALUES(1,'Dr. John Wilson','Cardiology');
INSERT INTO "Doctor" VALUES(2,'Dr. Jane Thompson','Gastroenterology');
INSERT INTO "Doctor" VALUES(3,'Dr. Tom Harris','Neurology');
INSERT INTO "Doctor" VALUES(4,'Dr. Emily Martin','Orthopedics');
INSERT INTO "Doctor" VALUES(5,'Dr. William Lee','Endocrinology');
CREATE TABLE Insurance_Plan(
    PlanID INTEGER PRIMARY KEY AUTOINCREMENT,
    PlanName TEXT,
    MonthlyPremium REAL);
INSERT INTO "Insurance_Plan" VALUES(1,'Basic',30.99);
INSERT INTO "Insurance_Plan" VALUES(2,'Standard',61.98);
INSERT INTO "Insurance_Plan" VALUES(3,'Deluxe',120.97);
INSERT INTO "Insurance_Plan" VALUES(4,'Premier',180.95);
INSERT INTO "Insurance_Plan" VALUES(5,'Ultimate',240.92);
CREATE TABLE Insurance_Plan_Coverage(
    PlanID INTEGER,
    CoverageID INTEGER,
    FOREIGN KEY(PlanID) REFERENCES Insurance_Plan(PlanID),
    FOREIGN KEY(CoverageID) REFERENCES Coverage(CoverageID),
    PRIMARY KEY(PlanID, CoverageID));
INSERT INTO "Insurance_Plan_Coverage" VALUES(1,1);
INSERT INTO "Insurance_Plan_Coverage" VALUES(2,1);
INSERT INTO "Insurance_Plan_Coverage" VALUES(3,1);
INSERT INTO "Insurance_Plan_Coverage" VALUES(4,1);
INSERT INTO "Insurance_Plan_Coverage" VALUES(5,1);
INSERT INTO "Insurance_Plan_Coverage" VALUES(5,2);
CREATE TABLE Patient(PatientID INTEGER PRIMARY KEY AUTOINCREMENT, Name TEXT, DOB TEXT);
INSERT INTO "Patient" VALUES(1,'John Doe','1990-01-01');
INSERT INTO "Patient" VALUES(2,'Jane Smith','1992-02-02');
INSERT INTO "Patient" VALUES(3,'Tom Johnson','1988-03-03');
INSERT INTO "Patient" VALUES(4,'Emily Brown','1994-04-04');
INSERT INTO "Patient" VALUES(5,'William Davis','1996-05-05');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Insurance_Plan',5);
INSERT INTO "sqlite_sequence" VALUES('Coverage',5);
INSERT INTO "sqlite_sequence" VALUES('Patient',5);
INSERT INTO "sqlite_sequence" VALUES('Doctor',5);
INSERT INTO "sqlite_sequence" VALUES('Appointment',5);
COMMIT;
