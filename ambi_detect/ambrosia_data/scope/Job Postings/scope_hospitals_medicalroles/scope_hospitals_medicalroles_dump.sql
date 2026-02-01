BEGIN TRANSACTION;
CREATE TABLE Hospitals (
    HospitalID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Hospitals" VALUES(1,'St. Mary''s General','New York');
INSERT INTO "Hospitals" VALUES(2,'Memorial Health System','Chicago');
INSERT INTO "Hospitals" VALUES(3,'University Hospital','Los Angeles');
INSERT INTO "Hospitals" VALUES(4,'Mount Sinai Medical Center','Miami');
INSERT INTO "Hospitals" VALUES(5,'Mercy Regional Medical Center','Phoenix');
CREATE TABLE Hospitals_Medical_Roles (
    HospitalID INTEGER,
    RoleID INTEGER,
    FOREIGN KEY (HospitalID) REFERENCES Hospitals(HospitalID),
    FOREIGN KEY (RoleID) REFERENCES MedicalRoles(RoleID),
    PRIMARY KEY (HospitalID, RoleID)
);
INSERT INTO "Hospitals_Medical_Roles" VALUES(1,1);
INSERT INTO "Hospitals_Medical_Roles" VALUES(2,1);
INSERT INTO "Hospitals_Medical_Roles" VALUES(3,1);
INSERT INTO "Hospitals_Medical_Roles" VALUES(4,1);
INSERT INTO "Hospitals_Medical_Roles" VALUES(5,1);
INSERT INTO "Hospitals_Medical_Roles" VALUES(5,4);
CREATE TABLE MedicalRoles (
    RoleID INTEGER PRIMARY KEY AUTOINCREMENT,
    RoleName VARCHAR(255),
    Salary REAL
);
INSERT INTO "MedicalRoles" VALUES(1,'Registered Nurse',70000.0);
INSERT INTO "MedicalRoles" VALUES(2,'Physician',130000.0);
INSERT INTO "MedicalRoles" VALUES(3,'Pharmacist',90000.0);
INSERT INTO "MedicalRoles" VALUES(4,'Radiologist',110000.0);
INSERT INTO "MedicalRoles" VALUES(5,'Lab Technician',65000.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hospitals',5);
INSERT INTO "sqlite_sequence" VALUES('MedicalRoles',5);
COMMIT;
