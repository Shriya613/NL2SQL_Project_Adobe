BEGIN TRANSACTION;
CREATE TABLE Appointments(
    AppointmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    CenterID INTEGER,
    PatientID INTEGER,
    AppointmentDate DATE,
    FOREIGN KEY(CenterID) REFERENCES Diagnostic_Centers(CenterID),
    FOREIGN KEY(PatientID) REFERENCES Patients(PatientID));
INSERT INTO "Appointments" VALUES(1,1,1,'2022-06-01');
INSERT INTO "Appointments" VALUES(2,2,2,'2022-06-02');
INSERT INTO "Appointments" VALUES(3,3,3,'2022-06-03');
INSERT INTO "Appointments" VALUES(4,4,4,'2022-06-04');
INSERT INTO "Appointments" VALUES(5,5,5,'2022-06-05');
CREATE TABLE Diagnostic_Centers(
    CenterID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT);
INSERT INTO "Diagnostic_Centers" VALUES(1,'Center A','123 Main St');
INSERT INTO "Diagnostic_Centers" VALUES(2,'Center B','456 Elm St');
INSERT INTO "Diagnostic_Centers" VALUES(3,'Center C','789 Oak St');
INSERT INTO "Diagnostic_Centers" VALUES(4,'Center D','01 Second Ave');
INSERT INTO "Diagnostic_Centers" VALUES(5,'Center E','Last St 23');
CREATE TABLE Diagnostic_Centers_Tests(
    DiagnosticCenterID INTEGER,
    TestTypeID INTEGER,
    FOREIGN KEY(DiagnosticCenterID) REFERENCES Diagnostic_Centers(CenterID),
    FOREIGN KEY(TestTypeID) REFERENCES Tests(TestTypeID),
    PRIMARY KEY(DiagnosticCenterID, TestTypeID));
INSERT INTO "Diagnostic_Centers_Tests" VALUES(1,1);
INSERT INTO "Diagnostic_Centers_Tests" VALUES(2,1);
INSERT INTO "Diagnostic_Centers_Tests" VALUES(3,1);
INSERT INTO "Diagnostic_Centers_Tests" VALUES(4,1);
INSERT INTO "Diagnostic_Centers_Tests" VALUES(5,1);
INSERT INTO "Diagnostic_Centers_Tests" VALUES(5,4);
CREATE TABLE Patients(
    PatientID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    PhoneNumber TEXT);
INSERT INTO "Patients" VALUES(1,'John','Doe','2000-01-01','555-123-4567');
INSERT INTO "Patients" VALUES(2,'Jane','Smith','1999-02-02','555-987-6543');
INSERT INTO "Patients" VALUES(3,'Alice','Brown','2001-03-03','555-000-0000');
INSERT INTO "Patients" VALUES(4,'Bob','White','1998-04-04','555-111-2222');
INSERT INTO "Patients" VALUES(5,'Emma','Green','2002-05-05','555-333-4444');
CREATE TABLE Results(
    ResultID INTEGER PRIMARY KEY AUTOINCREMENT,
    AppointmentID INTEGER,
    TestResultText TEXT,
    ResultDate DATE,
    FOREIGN KEY(AppointmentID) REFERENCES Appointments(AppointmentID));
INSERT INTO "Results" VALUES(1,1,'Normal','2022-06-01');
INSERT INTO "Results" VALUES(2,2,'Abnormal','2022-06-02');
INSERT INTO "Results" VALUES(3,3,'Positive','2022-06-03');
INSERT INTO "Results" VALUES(4,4,'Negative','2022-06-04');
INSERT INTO "Results" VALUES(5,5,'Inconclusive','2022-06-05');
CREATE TABLE Tests(
    TestTypeID INTEGER PRIMARY KEY AUTOINCREMENT,
    TestName TEXT);
INSERT INTO "Tests" VALUES(1,'Blood_Test');
INSERT INTO "Tests" VALUES(2,'X-Ray');
INSERT INTO "Tests" VALUES(3,'Ultrasound');
INSERT INTO "Tests" VALUES(4,'MRI');
INSERT INTO "Tests" VALUES(5,'CT Scan');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Diagnostic_Centers',5);
INSERT INTO "sqlite_sequence" VALUES('Tests',5);
INSERT INTO "sqlite_sequence" VALUES('Patients',5);
INSERT INTO "sqlite_sequence" VALUES('Appointments',5);
INSERT INTO "sqlite_sequence" VALUES('Results',5);
COMMIT;
