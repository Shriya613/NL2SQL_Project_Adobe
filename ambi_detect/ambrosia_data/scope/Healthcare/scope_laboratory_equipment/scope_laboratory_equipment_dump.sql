BEGIN TRANSACTION;
CREATE TABLE Appointments (
    appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER,
    patient_id INTEGER,
    date DATE,
    time TIME,
    FOREIGN KEY(doctor_id) REFERENCES Doctors(id),
    FOREIGN KEY(patient_id) REFERENCES Patients(id)
);
INSERT INTO "Appointments" VALUES(1,1,1,'2023-09-01','10:00');
INSERT INTO "Appointments" VALUES(2,2,2,'2023-09-02','11:00');
INSERT INTO "Appointments" VALUES(3,3,3,'2023-09-03','14:00');
INSERT INTO "Appointments" VALUES(4,4,4,'2023-09-04','15:00');
INSERT INTO "Appointments" VALUES(5,5,5,'2023-09-05','09:00');
CREATE TABLE Doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    specialization TEXT
);
INSERT INTO "Doctors" VALUES(1,'Dr. Smith','Cardiology');
INSERT INTO "Doctors" VALUES(2,'Dr. Johnson','Neurology');
INSERT INTO "Doctors" VALUES(3,'Dr. Brown','Pediatrics');
INSERT INTO "Doctors" VALUES(4,'Dr. Davis','Gastroenterology');
INSERT INTO "Doctors" VALUES(5,'Dr. Garcia','Orthopedics');
CREATE TABLE Equipment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    description TEXT
);
INSERT INTO "Equipment" VALUES(1,'Microscope','<description_for_microscope>');
INSERT INTO "Equipment" VALUES(2,'CT Scanner','<description_for_ct_scanner>');
INSERT INTO "Equipment" VALUES(3,'MRI Machine','<description_for_mri_machine>');
INSERT INTO "Equipment" VALUES(4,'X-Ray Machine','<description_for_xray_machine>');
INSERT INTO "Equipment" VALUES(5,'Ultrasound','<description_for_ultrasound>');
CREATE TABLE Laboratory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT
);
INSERT INTO "Laboratory" VALUES(1,'Central Lab','<address_of_central_lab>');
INSERT INTO "Laboratory" VALUES(2,'South Lab','<address_of_south_lab>');
INSERT INTO "Laboratory" VALUES(3,'East Lab','<address_of_east_lab>');
INSERT INTO "Laboratory" VALUES(4,'West Lab','<address_of_west_lab>');
INSERT INTO "Laboratory" VALUES(5,'North Lab','<address_of_north_lab>');
CREATE TABLE Laboratory_Equipment (
    laboratory_id INTEGER,
    equipment_type_id INTEGER,
    FOREIGN KEY(laboratory_id) REFERENCES Laboratory(id),
    FOREIGN KEY(equipment_type_id) REFERENCES Equipment(id),
    PRIMARY KEY(laboratory_id, equipment_type_id)
);
INSERT INTO "Laboratory_Equipment" VALUES(1,1);
INSERT INTO "Laboratory_Equipment" VALUES(2,1);
INSERT INTO "Laboratory_Equipment" VALUES(3,1);
INSERT INTO "Laboratory_Equipment" VALUES(4,1);
INSERT INTO "Laboratory_Equipment" VALUES(5,1);
INSERT INTO "Laboratory_Equipment" VALUES(5,4);
CREATE TABLE Patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender CHAR(1) 
);
INSERT INTO "Patients" VALUES(1,'John Doe',30,'M');
INSERT INTO "Patients" VALUES(2,'Jane Smith',28,'F');
INSERT INTO "Patients" VALUES(3,'Tom Johnson',45,'M');
INSERT INTO "Patients" VALUES(4,'Emily Williams',35,'F');
INSERT INTO "Patients" VALUES(5,'Robert Davis',60,'M');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Doctors',5);
INSERT INTO "sqlite_sequence" VALUES('Patients',5);
INSERT INTO "sqlite_sequence" VALUES('Appointments',5);
INSERT INTO "sqlite_sequence" VALUES('Equipment',5);
INSERT INTO "sqlite_sequence" VALUES('Laboratory',5);
COMMIT;
