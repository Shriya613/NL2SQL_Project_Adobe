BEGIN TRANSACTION;
CREATE TABLE Medications(
    medication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    dosage TEXT,
    side_effects TEXT);
INSERT INTO "Medications" VALUES(1,'Pain Relievers','As directed','Allergic reactions, stomach upset');
INSERT INTO "Medications" VALUES(2,'Anti-Inflammatories','As directed','Stomach bleeding, dizziness');
INSERT INTO "Medications" VALUES(3,'Respiratory Medications','As directed','Dry mouth, headache');
INSERT INTO "Medications" VALUES(4,'Cardiovascular Drugs','As directed','Dizziness, fatigue');
INSERT INTO "Medications" VALUES(5,'Gastrointestinal Medications','As directed','Nausea, diarrhea');
CREATE TABLE Pharmacies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Pharmacies" VALUES(1,'Ace Medical Center','123 Main St');
INSERT INTO "Pharmacies" VALUES(2,'Beechwood Pharmacy','456 Elm St');
INSERT INTO "Pharmacies" VALUES(3,'Cedar Health Solutions','789 Oak St');
INSERT INTO "Pharmacies" VALUES(4,'Dogwood Apothecary','098 Pine St');
INSERT INTO "Pharmacies" VALUES(5,'Evergreen Care Center','675 Maple Ave');
CREATE TABLE Pharmacy_Medications(
    pharmacy_medication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pharmacy_id INTEGER,
    medication_id INTEGER,
    FOREIGN KEY(pharmacy_id) REFERENCES Pharmacies(id),
    FOREIGN KEY(medication_id) REFERENCES Medications(medication_id));
INSERT INTO "Pharmacy_Medications" VALUES(1,1,1);
INSERT INTO "Pharmacy_Medications" VALUES(2,1,2);
INSERT INTO "Pharmacy_Medications" VALUES(3,2,1);
INSERT INTO "Pharmacy_Medications" VALUES(4,2,3);
INSERT INTO "Pharmacy_Medications" VALUES(5,3,1);
INSERT INTO "Pharmacy_Medications" VALUES(6,4,1);
INSERT INTO "Pharmacy_Medications" VALUES(7,5,1);
CREATE TABLE Pharmacy_Prescriptions(
    pharmacy_prescription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pharmacy_id INTEGER,
    prescription_id INTEGER,
    quantity INTEGER,
    FOREIGN KEY(pharmacy_id) REFERENCES Pharmacies(id),
    FOREIGN KEY(prescription_id) REFERENCES Prescriptions(prescription_id));
INSERT INTO "Pharmacy_Prescriptions" VALUES(1,1,1,10);
INSERT INTO "Pharmacy_Prescriptions" VALUES(2,1,2,15);
INSERT INTO "Pharmacy_Prescriptions" VALUES(3,2,3,20);
INSERT INTO "Pharmacy_Prescriptions" VALUES(4,2,4,25);
INSERT INTO "Pharmacy_Prescriptions" VALUES(5,3,5,30);
CREATE TABLE Prescriptions(
    prescription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pharmacy_id INTEGER,
    patient_name TEXT,
    date DATE,
    FOREIGN KEY(pharmacy_id) REFERENCES Pharmacies(id));
INSERT INTO "Prescriptions" VALUES(1,NULL,'John Doe','2021-10-01');
INSERT INTO "Prescriptions" VALUES(2,NULL,'Jane Smith','2021-10-02');
INSERT INTO "Prescriptions" VALUES(3,NULL,'Michael Johnson','2021-10-03');
INSERT INTO "Prescriptions" VALUES(4,NULL,'Emily Davis','2021-10-04');
INSERT INTO "Prescriptions" VALUES(5,NULL,'David Brown','2021-10-05');
INSERT INTO "Prescriptions" VALUES(6,1,'John Doe','2021-10-01');
INSERT INTO "Prescriptions" VALUES(7,1,'Jane Smith','2021-10-02');
INSERT INTO "Prescriptions" VALUES(8,2,'Michael Johnson','2021-10-03');
INSERT INTO "Prescriptions" VALUES(9,2,'Emily Davis','2021-10-04');
INSERT INTO "Prescriptions" VALUES(10,3,'David Brown','2021-10-05');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Pharmacies',5);
INSERT INTO "sqlite_sequence" VALUES('Medications',5);
INSERT INTO "sqlite_sequence" VALUES('Prescriptions',10);
INSERT INTO "sqlite_sequence" VALUES('Pharmacy_Medications',7);
INSERT INTO "sqlite_sequence" VALUES('Pharmacy_Prescriptions',5);
COMMIT;
