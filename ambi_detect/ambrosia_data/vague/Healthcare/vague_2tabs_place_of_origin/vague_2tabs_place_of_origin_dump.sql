BEGIN TRANSACTION;
CREATE TABLE Country (
    id INTEGER PRIMARY KEY,
    name TEXT,
    population INTEGER
);
INSERT INTO "Country" VALUES(1,'USA',328200000);
INSERT INTO "Country" VALUES(2,'Canada',37600000);
INSERT INTO "Country" VALUES(3,'UK',67100000);
INSERT INTO "Country" VALUES(4,'Germany',83000000);
INSERT INTO "Country" VALUES(5,'France',65200000);
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
CREATE TABLE Manufacturing_Company (
    id INTEGER PRIMARY KEY,
    company_name TEXT,
    address TEXT
);
INSERT INTO "Manufacturing_Company" VALUES(1,'ABC MedTech','123 Main St, New York, NY');
INSERT INTO "Manufacturing_Company" VALUES(2,'XYZ Biomedicals','456 Elm St, Chicago, IL');
INSERT INTO "Manufacturing_Company" VALUES(3,'MedPlus Innovations','789 Oak St, Los Angeles, CA');
INSERT INTO "Manufacturing_Company" VALUES(4,'HealthWave Solutions','101 Pine St, San Francisco, CA');
INSERT INTO "Manufacturing_Company" VALUES(5,'MediPure Technologies','111 Spruce St, Boston, MA');
CREATE TABLE Medical_Device (
    id INTEGER PRIMARY KEY,
    device_name TEXT,
    manufacturing_company_id INTEGER,
    country_id INTEGER,
    FOREIGN KEY(manufacturing_company_id) REFERENCES Manufacturing_Company(id),
    FOREIGN KEY(country_id) REFERENCES Country(id)
);
INSERT INTO "Medical_Device" VALUES(1,'Glucometer',1,1);
INSERT INTO "Medical_Device" VALUES(2,'EKG Machine',2,2);
INSERT INTO "Medical_Device" VALUES(3,'Blood Pressure Monitor',3,3);
INSERT INTO "Medical_Device" VALUES(4,'Insulin Pen',4,4);
INSERT INTO "Medical_Device" VALUES(5,'Oxygen Concentrator',5,5);
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
