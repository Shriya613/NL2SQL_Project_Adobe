BEGIN TRANSACTION;
CREATE TABLE Doctor (
    id INTEGER PRIMARY KEY,
    doctor_name TEXT,
    speciality TEXT
);
INSERT INTO "Doctor" VALUES(1,'Dr. Johnson','Cardiology');
INSERT INTO "Doctor" VALUES(2,'Dr. Brown','Endocrinology');
INSERT INTO "Doctor" VALUES(3,'Dr. White','Neurology');
INSERT INTO "Doctor" VALUES(4,'Dr. Black','Orthopedics');
INSERT INTO "Doctor" VALUES(5,'Dr. Green','Allergy and Immunology');
CREATE TABLE "Medical_Device" (
    id INTEGER PRIMARY KEY,
    device_name TEXT,
    country_id INTEGER,
    FOREIGN KEY(country_id) REFERENCES "Place_Of_Origin"(id)
);
INSERT INTO "Medical_Device" VALUES(1,'Glucometer',1);
INSERT INTO "Medical_Device" VALUES(2,'EKG Machine',2);
INSERT INTO "Medical_Device" VALUES(3,'Blood Pressure Monitor',3);
INSERT INTO "Medical_Device" VALUES(4,'Insulin Pen',4);
INSERT INTO "Medical_Device" VALUES(5,'Oxygen Concentrator',5);
CREATE TABLE Patient (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    age INTEGER
);
INSERT INTO "Patient" VALUES(1,'John','Doe',45);
INSERT INTO "Patient" VALUES(2,'Jane','Smith',32);
INSERT INTO "Patient" VALUES(3,'Alice','Johnson',28);
INSERT INTO "Patient" VALUES(4,'Bob','Williams',55);
INSERT INTO "Patient" VALUES(5,'Emily','Martin',23);
CREATE TABLE "Place_Of_Origin" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    population INTEGER
);
INSERT INTO "Place_Of_Origin" VALUES(1,'USA',328200000);
INSERT INTO "Place_Of_Origin" VALUES(2,'Canada',37600000);
INSERT INTO "Place_Of_Origin" VALUES(3,'UK',67100000);
INSERT INTO "Place_Of_Origin" VALUES(4,'Germany',83000000);
INSERT INTO "Place_Of_Origin" VALUES(5,'France',65200000);
CREATE TABLE Prescription (
    prescription_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    doctor_id INTEGER,
    medical_device_id INTEGER,
    quantity INTEGER,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES Patient(id),
    FOREIGN KEY(doctor_id) REFERENCES Doctor(id),
    FOREIGN KEY(medical_device_id) REFERENCES Medical_Device(id)
);
INSERT INTO "Prescription" VALUES(1,1,1,1,2,'2024-02-19 13:15:05');
INSERT INTO "Prescription" VALUES(2,2,2,2,3,'2024-02-19 13:15:05');
INSERT INTO "Prescription" VALUES(3,3,3,3,1,'2024-02-19 13:15:05');
INSERT INTO "Prescription" VALUES(4,4,4,4,2,'2024-02-19 13:15:05');
INSERT INTO "Prescription" VALUES(5,5,5,5,1,'2024-02-19 13:15:05');
COMMIT;
