BEGIN TRANSACTION;
CREATE TABLE Employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_name TEXT,
    position TEXT,
    healthcare_facility_id INTEGER,
    FOREIGN KEY(healthcare_facility_id) REFERENCES Healthcare_Facilities(id)
);
INSERT INTO "Employees" VALUES(1,'Emp 1','Doctor',1);
INSERT INTO "Employees" VALUES(2,'Emp 2','Nurse',1);
INSERT INTO "Employees" VALUES(3,'Emp 3','Receptionist',2);
INSERT INTO "Employees" VALUES(4,'Emp 4','Pharmacist',2);
INSERT INTO "Employees" VALUES(5,'Emp 5','Housekeeper',3);
CREATE TABLE "Healthcare_Facilities" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    founding_id INTEGER,
    FOREIGN KEY(founding_id) REFERENCES "Historical_Context"(id));
INSERT INTO "Healthcare_Facilities" VALUES(1,'Facility 1','Address 1',1);
INSERT INTO "Healthcare_Facilities" VALUES(2,'Facility 2','Address 2',2);
INSERT INTO "Healthcare_Facilities" VALUES(3,'Facility 3','Address 3',3);
INSERT INTO "Healthcare_Facilities" VALUES(4,'Facility 4','Address 4',4);
INSERT INTO "Healthcare_Facilities" VALUES(5,'Facility 5','Address 5',5);
CREATE TABLE "Historical_Context" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER
);
INSERT INTO "Historical_Context" VALUES(1,1980);
INSERT INTO "Historical_Context" VALUES(2,2003);
INSERT INTO "Historical_Context" VALUES(3,1965);
INSERT INTO "Historical_Context" VALUES(4,1978);
INSERT INTO "Historical_Context" VALUES(5,1992);
CREATE TABLE Patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    age INTEGER,
    healthcare_facility_id INTEGER,
    FOREIGN KEY(healthcare_facility_id) REFERENCES Healthcare_Facilities(id)
);
INSERT INTO "Patients" VALUES(1,'Patient 1',45,1);
INSERT INTO "Patients" VALUES(2,'Patient 2',60,1);
INSERT INTO "Patients" VALUES(3,'Patient 3',30,2);
INSERT INTO "Patients" VALUES(4,'Patient 4',55,2);
INSERT INTO "Patients" VALUES(5,'Patient 5',25,3);
CREATE TABLE Treatments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    treatment_name TEXT,
    start_date DATE,
    end_date DATE,
    patient_id INTEGER,
    healthcare_facility_id INTEGER,
    FOREIGN KEY(patient_id) REFERENCES Patients(id),
    FOREIGN KEY(healthcare_facility_id) REFERENCES Healthcare_Facilities(id)
);
INSERT INTO "Treatments" VALUES(1,'Surgery','2022-01-01','2022-01-05',1,1);
INSERT INTO "Treatments" VALUES(2,'Physiotherapy','2022-01-01','2022-01-31',2,1);
INSERT INTO "Treatments" VALUES(3,'Medication','2022-02-01','2022-02-28',3,2);
INSERT INTO "Treatments" VALUES(4,'Dialysis','2022-02-01','2022-02-28',4,2);
INSERT INTO "Treatments" VALUES(5,'Checkup','2022-03-01','2022-03-01',5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Historical_Context',5);
INSERT INTO "sqlite_sequence" VALUES('Employees',5);
INSERT INTO "sqlite_sequence" VALUES('Patients',5);
INSERT INTO "sqlite_sequence" VALUES('Treatments',5);
INSERT INTO "sqlite_sequence" VALUES('Healthcare_Facilities',5);
COMMIT;
