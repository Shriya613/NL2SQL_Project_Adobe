BEGIN TRANSACTION;
CREATE TABLE Appointments (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY(patient_id) REFERENCES Patients(id)
);
INSERT INTO "Appointments" VALUES(1,1,'2022-09-15 10:00:00','First session');
INSERT INTO "Appointments" VALUES(2,2,'2022-09-16 15:00:00','Follow-up');
INSERT INTO "Appointments" VALUES(3,3,'2022-09-17 10:30:00','Initial assessment');
INSERT INTO "Appointments" VALUES(4,4,'2022-09-18 14:00:00','Medication review');
INSERT INTO "Appointments" VALUES(5,5,'2022-09-19 11:00:00','Therapy session');
CREATE TABLE Clinics (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Clinics" VALUES(1,'Harmony Wellness Center');
INSERT INTO "Clinics" VALUES(2,'Sunny Hills Mental Health');
CREATE TABLE Patients (
    id INTEGER PRIMARY KEY,
    psychologist_id INTEGER,
    psychiatrist_id INTEGER,
    name TEXT,
    age INTEGER,
    diagnosis TEXT,
    FOREIGN KEY(psychologist_id) REFERENCES Psychologists(id),
    FOREIGN KEY(psychiatrist_id) REFERENCES Psychiatrists(id)
);
INSERT INTO "Patients" VALUES(1,1,2,'John Doe',30,'Anxiety Disorder');
INSERT INTO "Patients" VALUES(2,2,3,'Jane Roe',28,'Depression');
INSERT INTO "Patients" VALUES(3,1,4,'Michael Johnson',40,'Bipolar Disorder');
INSERT INTO "Patients" VALUES(4,2,1,'Emily Davis',35,'Schizophrenia');
INSERT INTO "Patients" VALUES(5,3,2,'David Wilson',25,'Autism Spectrum Disorder');
CREATE TABLE Psychiatrists (
    id INTEGER PRIMARY KEY,
    clinic_id INTEGER,
    name TEXT,
    specialty TEXT,
    FOREIGN KEY(clinic_id) REFERENCES Clinics(id)
);
INSERT INTO "Psychiatrists" VALUES(1,1,'Dr. Laura Black','Adult Psychiatry');
INSERT INTO "Psychiatrists" VALUES(2,2,'Dr. Carol Blue','Child and Adolescent Psychiatry');
INSERT INTO "Psychiatrists" VALUES(3,1,'Dr. Emily Gray','Addiction Medicine');
INSERT INTO "Psychiatrists" VALUES(4,2,'Dr. David Gold','Forensic Psychiatry');
CREATE TABLE Psychologists (
    id INTEGER PRIMARY KEY,
    clinic_id INTEGER,
    name TEXT,
    specialty TEXT,
    FOREIGN KEY(clinic_id) REFERENCES Clinics(id)
);
INSERT INTO "Psychologists" VALUES(1,1,'Dr. Jane Smith','Cognitive Behavioral Therapy');
INSERT INTO "Psychologists" VALUES(2,2,'Dr. John Brown','Existential Psychotherapy');
INSERT INTO "Psychologists" VALUES(3,1,'Dr. Alice Green','Gestalt Therapy');
INSERT INTO "Psychologists" VALUES(4,2,'Dr. Bob White','Dialectical Behavior Therapy');
CREATE TABLE TreatmentPlans (
    id INTEGER PRIMARY KEY,
    appointment_id INTEGER,
    plan TEXT,
    status TEXT,
    FOREIGN KEY(appointment_id) REFERENCES Appointments(id)
);
INSERT INTO "TreatmentPlans" VALUES(1,1,'Cognitive Behavioral Therapy plan','In progress');
INSERT INTO "TreatmentPlans" VALUES(2,2,'Existential Psychotherapy plan','Completed');
INSERT INTO "TreatmentPlans" VALUES(3,3,'Gestalt Therapy plan','Ongoing');
INSERT INTO "TreatmentPlans" VALUES(4,4,'Dialectical Behavior Therapy plan','Not started yet');
INSERT INTO "TreatmentPlans" VALUES(5,5,'Individual therapy plan','In progress');
COMMIT;
