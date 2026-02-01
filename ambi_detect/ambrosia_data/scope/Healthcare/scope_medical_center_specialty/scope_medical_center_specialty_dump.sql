BEGIN TRANSACTION;
CREATE TABLE Appointment (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Patient_ID INTEGER,
    Doctor_ID INTEGER,
    StartTime TIMESTAMP,
    EndTime TIMESTAMP,
    FOREIGN KEY(Patient_ID) REFERENCES Patient(ID),
    FOREIGN KEY(Doctor_ID) REFERENCES Doctor(ID));
INSERT INTO "Appointment" VALUES(1,1,1,'2022-01-01 09:00:00','2022-01-01 10:00:00');
INSERT INTO "Appointment" VALUES(2,2,2,'2022-01-02 09:00:00','2022-01-02 10:00:00');
INSERT INTO "Appointment" VALUES(3,3,3,'2022-01-03 09:00:00','2022-01-03 10:00:00');
INSERT INTO "Appointment" VALUES(4,4,4,'2022-01-04 09:00:00','2022-01-04 10:00:00');
INSERT INTO "Appointment" VALUES(5,5,5,'2022-01-05 09:00:00','2022-01-05 10:00:00');
CREATE TABLE Doctor (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    LicenseNumber TEXT);
INSERT INTO "Doctor" VALUES(1,'John','Doe','MD123456789');
INSERT INTO "Doctor" VALUES(2,'Jane','Smith','MD987654321');
INSERT INTO "Doctor" VALUES(3,'Michael','Johnson','MD345678901');
INSERT INTO "Doctor" VALUES(4,'Emily','Williams','MD789012345');
INSERT INTO "Doctor" VALUES(5,'David','Brown','MD234567890');
CREATE TABLE Medical_Center (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Medical_Center" VALUES(1,'St. Johns Hospital','New York');
INSERT INTO "Medical_Center" VALUES(2,'Mercy General Hospital','Chicago');
INSERT INTO "Medical_Center" VALUES(3,'Mount Sinai Hospital','Los Angeles');
INSERT INTO "Medical_Center" VALUES(4,'Memorial Hermann Hospital','Houston');
INSERT INTO "Medical_Center" VALUES(5,'Mayo Clinic','Rochester');
CREATE TABLE Medical_Center_Specialty (
    Medical_Center_ID INTEGER,
    Specialty_ID INTEGER,
    FOREIGN KEY(Medical_Center_ID) REFERENCES Medical_Center(ID),
    FOREIGN KEY(Specialty_ID) REFERENCES Specialty(ID));
INSERT INTO "Medical_Center_Specialty" VALUES(1,1);
INSERT INTO "Medical_Center_Specialty" VALUES(2,1);
INSERT INTO "Medical_Center_Specialty" VALUES(3,1);
INSERT INTO "Medical_Center_Specialty" VALUES(4,1);
INSERT INTO "Medical_Center_Specialty" VALUES(5,1);
INSERT INTO "Medical_Center_Specialty" VALUES(5,2);
CREATE TABLE Patient (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    PhoneNumber CHAR(10));
INSERT INTO "Patient" VALUES(1,'Alice','Wilson','1990-01-01','555-123-4567');
INSERT INTO "Patient" VALUES(2,'Bob','Thompson','1985-02-02','555-987-6543');
INSERT INTO "Patient" VALUES(3,'Carol','Jackson','1975-03-03','555-432-1234');
INSERT INTO "Patient" VALUES(4,'Daniel','Martin','1992-04-04','555-654-3210');
INSERT INTO "Patient" VALUES(5,'Elizabeth','Taylor','1988-05-05','555-132-4568');
CREATE TABLE Specialty (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT);
INSERT INTO "Specialty" VALUES(1,'Emergency Medicine');
INSERT INTO "Specialty" VALUES(2,'Cardiology');
INSERT INTO "Specialty" VALUES(3,'Gastroenterology');
INSERT INTO "Specialty" VALUES(4,'Neurology');
INSERT INTO "Specialty" VALUES(5,'Orthopedics');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Medical_Center',5);
INSERT INTO "sqlite_sequence" VALUES('Specialty',5);
INSERT INTO "sqlite_sequence" VALUES('Doctor',5);
INSERT INTO "sqlite_sequence" VALUES('Patient',5);
INSERT INTO "sqlite_sequence" VALUES('Appointment',5);
COMMIT;
