BEGIN TRANSACTION;
CREATE TABLE Doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    specialty TEXT,
    hospital_id INTEGER,
    FOREIGN KEY(hospital_id) REFERENCES Hospitals(id)
);
INSERT INTO "Doctors" VALUES(1,'Dr. John Smith','Cardiology',1);
INSERT INTO "Doctors" VALUES(2,'Dr. Jane Doe','General Practice',2);
INSERT INTO "Doctors" VALUES(3,'Dr. Mike Johnson','Orthopedics',3);
INSERT INTO "Doctors" VALUES(4,'Dr. Sarah Brown','Pediatrics',4);
INSERT INTO "Doctors" VALUES(5,'Dr. Emily Davis','Neurology',5);
CREATE TABLE Hospitals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city TEXT,
    neighborhood TEXT,
    phone_number TEXT UNIQUE
);
INSERT INTO "Hospitals" VALUES(1,'St. Mary''s Hospital','New York City','Midtown','+1-212-500-1111');
INSERT INTO "Hospitals" VALUES(2,'Mount Sinai Medical Center','Miami','Downtown','+1-305-555-2222');
INSERT INTO "Hospitals" VALUES(3,'Memorial Hermann Hospital','Houston','Medical Center','+1-713-244-6000');
INSERT INTO "Hospitals" VALUES(4,'Cleveland Clinic','Cleveland','Gates Mills Boulevard','+1-216-444-1000');
INSERT INTO "Hospitals" VALUES(5,'Mayo Clinic','Rochester','First Street SW','+1-507-538-0000');
CREATE TABLE InsurancePlans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_name TEXT,
    premium INTEGER,
    coverage_details TEXT,
    provider_id INTEGER,
    FOREIGN KEY(provider_id) REFERENCES Providers(id)
);
INSERT INTO "InsurancePlans" VALUES(1,'Basic Plan',100,'Covers essential medical services',1);
INSERT INTO "InsurancePlans" VALUES(2,'Standard Plan',200,'Includes preventive care and routine checkups',2);
INSERT INTO "InsurancePlans" VALUES(3,'Deluxe Plan',300,'Comprehensive coverage with no out-of-pocket costs',3);
INSERT INTO "InsurancePlans" VALUES(4,'Ultimate Plan',400,'Most extensive coverage available',4);
INSERT INTO "InsurancePlans" VALUES(5,'VIP Plan',500,'Exclusive benefits and concierge service',5);
CREATE TABLE Patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE,
    insurance_plan_id INTEGER,
    FOREIGN KEY(insurance_plan_id) REFERENCES InsurancePlans(id)
);
INSERT INTO "Patients" VALUES(1,'John','Doe','1980-01-01',1);
INSERT INTO "Patients" VALUES(2,'Jane','Smith','1985-02-02',2);
INSERT INTO "Patients" VALUES(3,'Emily','Brown','1990-03-03',3);
INSERT INTO "Patients" VALUES(4,'Mike','Johnson','1995-04-04',4);
INSERT INTO "Patients" VALUES(5,'Sarah','Davis','2000-05-05',5);
CREATE TABLE Prescriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    doctor_id INTEGER,
    medication TEXT,
    dosage TEXT,
    quantity INTEGER,
    refillable BOOLEAN,
    prescription_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES Patients(id),
    FOREIGN KEY(doctor_id) REFERENCES Doctors(id)
);
INSERT INTO "Prescriptions" VALUES(1,1,1,'Lisinopril','5mg',30,1,'2021-01-01 10:00:00');
INSERT INTO "Prescriptions" VALUES(2,2,2,'Metformin','500mg',30,0,'2021-02-01 11:00:00');
INSERT INTO "Prescriptions" VALUES(3,3,3,'Codeine','60mg',30,1,'2021-03-01 12:00:00');
INSERT INTO "Prescriptions" VALUES(4,4,4,'Albuterol','9mg',30,1,'2021-04-01 13:00:00');
INSERT INTO "Prescriptions" VALUES(5,5,5,'Azathioprine','50mg',30,0,'2021-05-01 14:00:00');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Doctors',5);
INSERT INTO "sqlite_sequence" VALUES('Hospitals',5);
INSERT INTO "sqlite_sequence" VALUES('InsurancePlans',5);
INSERT INTO "sqlite_sequence" VALUES('Patients',5);
INSERT INTO "sqlite_sequence" VALUES('Prescriptions',5);
COMMIT;
