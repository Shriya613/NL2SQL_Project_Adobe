BEGIN TRANSACTION;
CREATE TABLE Departments(
    DepartmentID INTEGER PRIMARY KEY,
    DepartmentName TEXT,
    HospitalID INTEGER,
    FOREIGN KEY (HospitalID) REFERENCES Hospitals(HospitalID));
INSERT INTO "Departments" VALUES(1,'Cardiac Surgery',1);
INSERT INTO "Departments" VALUES(2,'Neurosurgery',2);
INSERT INTO "Departments" VALUES(3,'Orthopedics',3);
INSERT INTO "Departments" VALUES(4,'Pediatrics',4);
INSERT INTO "Departments" VALUES(5,'Emergency Medicine',5);
CREATE TABLE Employees(
    EmployeeID INTEGER PRIMARY KEY,
    Email TEXT,
    PhoneNumber TEXT,
    MedicalProfession TEXT, --This field will contain Nurse, Physicians Assistants and others.
    Patient TEXT --This field will contain patient names like Anne Buche and others.
, FullName TEXT);
INSERT INTO "Employees" VALUES(1,'jane.doe@example.com','555-1234','Nurse','Anne Buche','Jane Doe');
INSERT INTO "Employees" VALUES(2,'mike.smith@example.com','555-5678','Nurse','Alice Johnson','Mike Smith');
INSERT INTO "Employees" VALUES(3,'linda.williams@example.com','555-9012','Physician''s Assistant','Anne Buche','Linda Williams');
INSERT INTO "Employees" VALUES(4,'tom.brown@example.com','555-1234','Physician''s Assistant','David Lee','Tom Brown');
INSERT INTO "Employees" VALUES(5,'karen.green@example.com','555-9876','Doctor',NULL,'Karen Green');
CREATE TABLE Hospitals(
    HospitalID INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT);
INSERT INTO "Hospitals" VALUES(1,'St. Mary''s Hospital','New York');
INSERT INTO "Hospitals" VALUES(2,'Mount Sinai Hospital','Chicago');
INSERT INTO "Hospitals" VALUES(3,'Rush University Medical Center','Chicago');
INSERT INTO "Hospitals" VALUES(4,'Yale New Haven Health','Connecticut');
INSERT INTO "Hospitals" VALUES(5,'John Hopkins Medicine','Baltimore');
CREATE TABLE JobPostings(
    JobPostID INTEGER PRIMARY KEY,
    PositionTitle TEXT,
    Description TEXT,
    Requirements TEXT,
    StartDate DATE,
    EndDate DATE,
    Salary DECIMAL(10,2),
    Vacancy INT ,
    DepartmentID INTEGER,
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID));
INSERT INTO "JobPostings" VALUES(1,'Cardiovascular Nurse','Experience in cardiac care.','BLS certification required.','2023-01-01','2023-12-31',65000,3,1);
INSERT INTO "JobPostings" VALUES(2,'Neurology Nurse','Neurosurgical experience necessary.','ACLS certification required.','2023-02-01','2023-12-31',60000,2,2);
INSERT INTO "JobPostings" VALUES(3,'Orthopaedic PA','Assisting with orthopaedic surgeries.','Certified PAs only.','2023-03-01','2023-12-31',65000,1,3);
INSERT INTO "JobPostings" VALUES(4,'Pediatric Nurse','Pediatric nursing experience required.','CPR certified.','2023-04-01','2023-12-31',60000,4,4);
INSERT INTO "JobPostings" VALUES(5,'EM Resident','ER experience essential.','MD degree required.','2023-05-01','2023-12-31',55000,5,5);
COMMIT;
