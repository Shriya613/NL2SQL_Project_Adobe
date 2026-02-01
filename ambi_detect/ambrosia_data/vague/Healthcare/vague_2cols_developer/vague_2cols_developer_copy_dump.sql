BEGIN TRANSACTION;
CREATE TABLE Appointments (
    appointmentId INTEGER PRIMARY KEY,
    doctorId INTEGER,
    patientId INTEGER,
    date TIMESTAMP,
    FOREIGN KEY(doctorId) REFERENCES Doctors(id),
    FOREIGN KEY(patientId) REFERENCES Patients(patientId)
);
INSERT INTO "Appointments" VALUES(1,1,1,'2022-01-01 10:00:00');
INSERT INTO "Appointments" VALUES(2,2,2,'2022-02-02 14:00:00');
CREATE TABLE Companies (
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT
);
INSERT INTO "Companies" VALUES(1,'MedTech Inc.','New York, USA');
INSERT INTO "Companies" VALUES(2,'Health Innovations Ltd.','London, UK');
CREATE TABLE "DeviceDevelopment" (
    Developer INTEGER,
    medicalDeviceId INTEGER,
    startDate DATE,
    endDate DATE,
    FOREIGN KEY(Developer) REFERENCES Scientists(id),
    FOREIGN KEY(Developer) REFERENCES Companies(id),
    FOREIGN KEY(medicalDeviceId) REFERENCES MedicalDevices(id),
    PRIMARY KEY(Developer, medicalDeviceId)
);
INSERT INTO "DeviceDevelopment" VALUES(1,1,'2021-01-01','2021-12-31');
INSERT INTO "DeviceDevelopment" VALUES(2,2,'2020-01-01','2020-12-31');
CREATE TABLE Doctors (
    doctorId INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    specialty TEXT
);
INSERT INTO "Doctors" VALUES(1,'Dr. Alice','Brown','Cardiology');
INSERT INTO "Doctors" VALUES(2,'Dr. Charles','Green','Neurology');
CREATE TABLE MedicalDevices (
    id INTEGER PRIMARY KEY,
    deviceName TEXT,
    description TEXT,
    price REAL
);
INSERT INTO "MedicalDevices" VALUES(1,'Glucose Monitor','Monitors blood glucose levels',99.99);
INSERT INTO "MedicalDevices" VALUES(2,'Heart Rate Sensor','Measures heart rate',69.99);
CREATE TABLE Patients (
    patientId INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE,
    gender CHAR(1) 
);
INSERT INTO "Patients" VALUES(1,'Michael','Johnson','1990-01-01','M');
INSERT INTO "Patients" VALUES(2,'Emily','Williams','1995-02-02','F');
CREATE TABLE Scientists (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    birthDate DATE
);
INSERT INTO "Scientists" VALUES(1,'John','Smith','1970-01-01');
INSERT INTO "Scientists" VALUES(2,'Jane','Doe','1980-02-02');
COMMIT;
