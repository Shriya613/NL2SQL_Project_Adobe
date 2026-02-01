BEGIN TRANSACTION;
CREATE TABLE Diagnosis (
    diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    diagnosis_text TEXT,
    diagnosis_code TEXT,
    FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
);
INSERT INTO "Diagnosis" VALUES(1,1,'Hypertension','I10.9');
INSERT INTO "Diagnosis" VALUES(2,2,'Type 2 diabetes mellitus','E11.9');
CREATE TABLE DoctorPatient (
    doctor_patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER,
    patient_id INTEGER,
    relationship TEXT,
    FOREIGN KEY (doctor_id) REFERENCES Doctors(doctor_id),
    FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
);
INSERT INTO "DoctorPatient" VALUES(1,1,1,'Primary Care Physician');
INSERT INTO "DoctorPatient" VALUES(2,2,2,'Specialist Referral');
CREATE TABLE Doctors (
    doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    specialty TEXT
);
INSERT INTO "Doctors" VALUES(1,'Dr. John Doe','Cardiology');
INSERT INTO "Doctors" VALUES(2,'Dr. Jane Smith','Neurology');
INSERT INTO "Doctors" VALUES(3,'Dr. Tom Williams','Psychiatry');
INSERT INTO "Doctors" VALUES(4,'Dr. Emily Jones','Endocrinology');
INSERT INTO "Doctors" VALUES(5,'Dr. Michael Taylor','Gastroenterology');
CREATE TABLE Medications (
    medication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    antibiotic TEXT,
    common_side_effects TEXT,
    serious_side_effects TEXT,
    dosage TEXT
);
INSERT INTO "Medications" VALUES(1,'Amoxicillin','Diarrhea, nausea','Rash, liver damage','500mg every 12 hours');
INSERT INTO "Medications" VALUES(2,'Ciprofloxacin','Nausea, diarrhea','Tendonitis, allergies','500mg twice daily');
INSERT INTO "Medications" VALUES(3,'Azithromycin','Headache, dizziness','Hearing loss, liver damage','500mg once daily');
INSERT INTO "Medications" VALUES(4,'Levofloxacin','Dizziness, headache','Allergies, tendonitis','750mg once daily');
INSERT INTO "Medications" VALUES(5,'Metronidazole','Nausea, vomiting','Peripheral neuropathy, seizures','500mg three times a day');
CREATE TABLE Patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER 
);
INSERT INTO "Patients" VALUES(1,'John Smith',30);
INSERT INTO "Patients" VALUES(2,'Jane Doe',28);
INSERT INTO "Patients" VALUES(3,'Tom Brown',40);
INSERT INTO "Patients" VALUES(4,'Emily White',26);
INSERT INTO "Patients" VALUES(5,'Michael Johnson',32);
CREATE TABLE Prescriptions (
    prescription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    doctor_id INTEGER,
    medication_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (patient_id) REFERENCES Patients(patient_id),
    FOREIGN KEY (doctor_id) REFERENCES Doctors(doctor_id),
    FOREIGN KEY (medication_id) REFERENCES Medications(medication_id)
);
INSERT INTO "Prescriptions" VALUES(1,1,1,1,'2023-01-01','2023-01-10');
INSERT INTO "Prescriptions" VALUES(2,2,2,2,'2023-01-15','2023-01-25');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Patients',5);
INSERT INTO "sqlite_sequence" VALUES('Doctors',5);
INSERT INTO "sqlite_sequence" VALUES('Medications',5);
INSERT INTO "sqlite_sequence" VALUES('Prescriptions',2);
INSERT INTO "sqlite_sequence" VALUES('DoctorPatient',2);
INSERT INTO "sqlite_sequence" VALUES('Diagnosis',2);
COMMIT;
