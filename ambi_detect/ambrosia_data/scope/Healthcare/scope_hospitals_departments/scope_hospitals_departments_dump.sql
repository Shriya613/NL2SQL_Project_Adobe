BEGIN TRANSACTION;
CREATE TABLE Appointments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patientId INTEGER,
    doctorId INTEGER,
    date DATETIME,
    time TEXT,
    reasonForVisit TEXT,
    FOREIGN KEY(patientId) REFERENCES Patients(id),
    FOREIGN KEY(doctorId) REFERENCES Doctors(id)
);
INSERT INTO "Appointments" VALUES(1,1,1,'2022-01-01 10:00:00','10:00 AM','Influenza A');
INSERT INTO "Appointments" VALUES(2,2,2,'2022-01-02 02:00:00 PM','02:00 PM','Fractured wrist');
INSERT INTO "Appointments" VALUES(3,3,3,'2022-01-03 03:00:00 PM','03:00 PM','Panic attack');
INSERT INTO "Appointments" VALUES(4,4,4,'2022-01-04 04:00:00 PM','04:00 PM','Shortness of breath');
INSERT INTO "Appointments" VALUES(5,5,5,'2022-01-05 05:00:00 PM','05:00 PM','Allergies');
CREATE TABLE Departments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    departmentName TEXT
);
INSERT INTO "Departments" VALUES(1,'Cardiology');
INSERT INTO "Departments" VALUES(2,'Neurology');
INSERT INTO "Departments" VALUES(3,'Gastroenterology');
INSERT INTO "Departments" VALUES(4,'Orthopedics');
INSERT INTO "Departments" VALUES(5,'Pulmonary Medicine');
CREATE TABLE Doctors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    specialtyId INTEGER,
    hospitalId INTEGER,
    FOREIGN KEY(specialtyId) REFERENCES Departments(id),
    FOREIGN KEY(hospitalId) REFERENCES Hospitals(id)
);
INSERT INTO "Doctors" VALUES(1,'Dr. Smith',1,1);
INSERT INTO "Doctors" VALUES(2,'Dr. Johnson',2,1);
INSERT INTO "Doctors" VALUES(3,'Dr. Brown',3,2);
INSERT INTO "Doctors" VALUES(4,'Dr. Davis',4,2);
INSERT INTO "Doctors" VALUES(5,'Dr. Miller',5,3);
CREATE TABLE Hospitals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Hospitals" VALUES(1,'St. Marys Hospital','123 Main St');
INSERT INTO "Hospitals" VALUES(2,'New York General','456 Broadway St');
INSERT INTO "Hospitals" VALUES(3,'Metro Medical Center','789 Central Ave');
INSERT INTO "Hospitals" VALUES(4,'Cedars Sinai','1011 Ocean Blvd');
INSERT INTO "Hospitals" VALUES(5,'John Hopkins','2020 University Pl');
CREATE TABLE Hospitals_Departments(
    hospitalId INTEGER,
    departmentId INTEGER,
    PRIMARY KEY(hospitalId, departmentId),
    FOREIGN KEY(hospitalId) REFERENCES Hospitals(id),
    FOREIGN KEY(departmentId) REFERENCES Departments(id)
);
INSERT INTO "Hospitals_Departments" VALUES(1,1);
INSERT INTO "Hospitals_Departments" VALUES(1,2);
INSERT INTO "Hospitals_Departments" VALUES(2,1);
INSERT INTO "Hospitals_Departments" VALUES(2,3);
INSERT INTO "Hospitals_Departments" VALUES(3,4);
INSERT INTO "Hospitals_Departments" VALUES(3,5);
INSERT INTO "Hospitals_Departments" VALUES(3,1);
INSERT INTO "Hospitals_Departments" VALUES(4,1);
INSERT INTO "Hospitals_Departments" VALUES(5,1);
CREATE TABLE Patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    age INTEGER ,
    illnessText TEXT
);
INSERT INTO "Patients" VALUES(1,'Jane','Doe',35,'Flu');
INSERT INTO "Patients" VALUES(2,'Michael','Smith',45,'Broken Arm');
INSERT INTO "Patients" VALUES(3,'Emily','Johnson',25,'Anxiety');
INSERT INTO "Patients" VALUES(4,'David','Brown',65,'Diabetes');
INSERT INTO "Patients" VALUES(5,'Karen','Miller',55,'Asthma');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hospitals',5);
INSERT INTO "sqlite_sequence" VALUES('Departments',5);
INSERT INTO "sqlite_sequence" VALUES('Doctors',5);
INSERT INTO "sqlite_sequence" VALUES('Patients',5);
INSERT INTO "sqlite_sequence" VALUES('Appointments',5);
COMMIT;
