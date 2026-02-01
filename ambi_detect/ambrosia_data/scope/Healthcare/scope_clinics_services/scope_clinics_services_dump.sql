BEGIN TRANSACTION;
CREATE TABLE Appointments (
    appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    clinic_id INTEGER,
    service_id INTEGER,
    appointment_date TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES Patients(patient_id),
    FOREIGN KEY(clinic_id) REFERENCES Clinics(clinic_id),
    FOREIGN KEY(service_id) REFERENCES Services(service_id)
);
INSERT INTO "Appointments" VALUES(1,1,1,1,'2022-01-10 10:00:00');
INSERT INTO "Appointments" VALUES(2,2,2,3,'2022-01-12 14:00:00');
INSERT INTO "Appointments" VALUES(3,3,3,5,'2022-01-14 09:00:00');
INSERT INTO "Appointments" VALUES(4,4,4,1,'2022-01-16 11:00:00');
INSERT INTO "Appointments" VALUES(5,5,5,4,'2022-01-18 15:00:00');
CREATE TABLE Clinics (
    clinic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT
);
INSERT INTO "Clinics" VALUES(1,'Healthy Living','123 Main St');
INSERT INTO "Clinics" VALUES(2,'Medical Center','456 Oak Ave');
INSERT INTO "Clinics" VALUES(3,'Family Care','789 Pine Blvd');
INSERT INTO "Clinics" VALUES(4,'Northside Hospital','101 Elm Rd');
INSERT INTO "Clinics" VALUES(5,'Southside Medical','202 Willow Ln');
CREATE TABLE Clinics_Services (
    clinics_services_id INTEGER PRIMARY KEY AUTOINCREMENT,
    clinic_id INTEGER,
    service_id INTEGER,
    FOREIGN KEY(clinic_id) REFERENCES Clinics(clinic_id),
    FOREIGN KEY(service_id) REFERENCES Services(service_id)
);
INSERT INTO "Clinics_Services" VALUES(1,1,1);
INSERT INTO "Clinics_Services" VALUES(2,2,1);
INSERT INTO "Clinics_Services" VALUES(3,3,1);
INSERT INTO "Clinics_Services" VALUES(4,4,1);
INSERT INTO "Clinics_Services" VALUES(5,5,1);
INSERT INTO "Clinics_Services" VALUES(6,5,5);
CREATE TABLE Patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    phone_number VARCHAR(15)
);
INSERT INTO "Patients" VALUES(1,'John','Smith','1985-05-15','555-123-4567');
INSERT INTO "Patients" VALUES(2,'Jane','Doe','1990-08-20','555-987-6543');
INSERT INTO "Patients" VALUES(3,'Michael','Brown','1988-11-22','555-345-7890');
INSERT INTO "Patients" VALUES(4,'Emily','White','1992-03-01','555-135-7902');
INSERT INTO "Patients" VALUES(5,'David','Black','1986-07-14','555-567-8903');
CREATE TABLE Services (
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT
);
INSERT INTO "Services" VALUES(1,'Vaccination');
INSERT INTO "Services" VALUES(2,'Physical Examination');
INSERT INTO "Services" VALUES(3,'Dental Checkup');
INSERT INTO "Services" VALUES(4,'Laboratory Tests');
INSERT INTO "Services" VALUES(5,'X-ray Imaging');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Clinics',5);
INSERT INTO "sqlite_sequence" VALUES('Services',5);
INSERT INTO "sqlite_sequence" VALUES('Patients',5);
INSERT INTO "sqlite_sequence" VALUES('Appointments',5);
INSERT INTO "sqlite_sequence" VALUES('Clinics_Services',6);
COMMIT;
