BEGIN TRANSACTION;
CREATE TABLE Appointment (
    appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TIMESTAMP,
    patient_id INTEGER,
    doctor_id INTEGER,
    medicalrecord_id INTEGER,
    FOREIGN KEY(patient_id) REFERENCES Patient(patient_id),
    FOREIGN KEY(doctor_id) REFERENCES Doctor(doctor_id),
    FOREIGN KEY(medicalrecord_id) REFERENCES MedicalRecord(medicalrecord_id)
);
INSERT INTO "Appointment" VALUES(1,'2022-10-12 10:00:00',1,1,1);
INSERT INTO "Appointment" VALUES(2,'2022-10-13 14:00:00',2,2,2);
INSERT INTO "Appointment" VALUES(3,'2022-10-14 16:00:00',3,3,3);
INSERT INTO "Appointment" VALUES(4,'2022-10-15 18:00:00',4,4,4);
INSERT INTO "Appointment" VALUES(5,'2022-10-16 20:00:00',5,5,5);
CREATE TABLE Doctor (
    doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    specialty TEXT
);
INSERT INTO "Doctor" VALUES(1,'Dr.','Jones','Cardiology');
INSERT INTO "Doctor" VALUES(2,'Dr.','Garcia','Endocrinology');
INSERT INTO "Doctor" VALUES(3,'Dr.','Lee','Pediatrics');
INSERT INTO "Doctor" VALUES(4,'Dr.','Taylor','Neurology');
INSERT INTO "Doctor" VALUES(5,'Dr.','Adams','Nephrology');
CREATE TABLE MedicalRecord (
    medicalrecord_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    diagnosis TEXT,
    treatment_plan TEXT,
    FOREIGN KEY(patient_id) REFERENCES Patient(patient_id)
);
INSERT INTO "MedicalRecord" VALUES(1,1,'Type 2 Diabetes','Diet and Exercise');
INSERT INTO "MedicalRecord" VALUES(2,2,'Epilepsy','Antiepileptic Drugs');
INSERT INTO "MedicalRecord" VALUES(3,3,'Anxiety Disorder','Counseling and Medication');
INSERT INTO "MedicalRecord" VALUES(4,4,'Depression','Antidepressants');
INSERT INTO "MedicalRecord" VALUES(5,5,'Asthma','Inhalers');
CREATE TABLE "Patient" (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE,
    phone_number VARCHAR(15),
    age_group_id INTEGER,
    FOREIGN KEY(age_group_id) REFERENCES "target"(id)
);
INSERT INTO "Patient" VALUES(1,'John','Doe','1995-06-15','555-123-4567',3);
INSERT INTO "Patient" VALUES(2,'Jane','Smith','2000-07-20','555-987-6543',5);
INSERT INTO "Patient" VALUES(3,'Michael','Brown','1990-08-25','555-555-5555',4);
INSERT INTO "Patient" VALUES(4,'Emily','Green','1988-09-15','555-333-3333',2);
INSERT INTO "Patient" VALUES(5,'William','Black','1975-10-20','555-222-2222',6);
CREATE TABLE "Vaccine" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vaccine_name TEXT,
    age_group_id INTEGER,
    FOREIGN KEY(age_group_id) REFERENCES "target"(id));
INSERT INTO "Vaccine" VALUES(1,'Flu',2);
INSERT INTO "Vaccine" VALUES(2,'Pneumonia',3);
INSERT INTO "Vaccine" VALUES(3,'Hepatitis B',1);
INSERT INTO "Vaccine" VALUES(4,'Meningococcal',4);
INSERT INTO "Vaccine" VALUES(5,'Varicella',1);
CREATE TABLE "target" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "target" VALUES(1,'0-4');
INSERT INTO "target" VALUES(2,'5-9');
INSERT INTO "target" VALUES(3,'10-13');
INSERT INTO "target" VALUES(4,'14-17');
INSERT INTO "target" VALUES(5,'18+');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('target',5);
INSERT INTO "sqlite_sequence" VALUES('Doctor',5);
INSERT INTO "sqlite_sequence" VALUES('MedicalRecord',5);
INSERT INTO "sqlite_sequence" VALUES('Appointment',5);
INSERT INTO "sqlite_sequence" VALUES('Vaccine',5);
INSERT INTO "sqlite_sequence" VALUES('Patient',5);
COMMIT;
