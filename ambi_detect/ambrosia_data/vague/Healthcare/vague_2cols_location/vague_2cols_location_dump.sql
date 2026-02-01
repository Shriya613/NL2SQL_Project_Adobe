BEGIN TRANSACTION;
CREATE TABLE Appointments (
    appointment_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    doctor_id INTEGER,
    date TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES Patients(patient_id),
    FOREIGN KEY(doctor_id) REFERENCES Doctors(doctor_id)
);
CREATE TABLE Doctors (
    doctor_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    specialty TEXT,
    hospital_id INTEGER,
    FOREIGN KEY(hospital_id) REFERENCES Hospitals(hospital_id)
);
CREATE TABLE Hospitals (
    hospital_id INTEGER PRIMARY KEY,
    name TEXT,
    phone_number CHAR(10),
    city TEXT,
    state TEXT
);
INSERT INTO "Hospitals" VALUES(1,'St. Mary''s General','123-456-7890','San Francisco','SF');
INSERT INTO "Hospitals" VALUES(2,'Westside Medical Center','987-654-3210','Portland','OR');
INSERT INTO "Hospitals" VALUES(3,'Central Valley Regional','555-555-5555','Fresno','CA');
INSERT INTO "Hospitals" VALUES(4,'South Coast Community','111-222-3333','Long Beach','LA');
INSERT INTO "Hospitals" VALUES(5,'Eastern Seaboard University','444-444-4444','Boston','MA');
CREATE TABLE Patients (
    patient_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE,
    city TEXT,
    state TEXT
);
INSERT INTO "Patients" VALUES(1,'John','Doe','2000-08-30','New York','NY');
INSERT INTO "Patients" VALUES(2,'Jane','Smith','1995-06-25','Los Angeles','CA');
INSERT INTO "Patients" VALUES(3,'Michael','Brown','1992-03-17','Chicago','IL');
INSERT INTO "Patients" VALUES(4,'Emily','White','1998-11-12','Houston','TX');
INSERT INTO "Patients" VALUES(5,'William','Black','2001-07-03','Phoenix','AZ');
COMMIT;
