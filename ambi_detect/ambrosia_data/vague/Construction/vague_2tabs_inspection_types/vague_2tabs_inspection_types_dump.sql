BEGIN TRANSACTION;
CREATE TABLE Employee (
    employeeID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    role TEXT,
    hireDate DATE,
    supervisorID INTEGER,
    FOREIGN KEY(supervisorID) REFERENCES Employee(employeeID)
);
INSERT INTO "Employee" VALUES(1,'John','Doe','Structural Engineer','2022-01-01',NULL);
INSERT INTO "Employee" VALUES(2,'Jane','Smith','Safety Officer','2022-01-01',NULL);
INSERT INTO "Employee" VALUES(3,'Mike','Johnson','Site Supervisor','2022-01-01',1);
INSERT INTO "Employee" VALUES(4,'Anna','Williams','Architect','2022-01-01',1);
CREATE TABLE Material (
    materialID INTEGER PRIMARY KEY,
    type TEXT,
    quantity INTEGER,
    unit TEXT,
    price DECIMAL(10,2),
    projectID INTEGER,
    FOREIGN KEY(projectID) REFERENCES Project(projectID)
);
INSERT INTO "Material" VALUES(1,'Concrete',100,'cubic yards',75,1);
INSERT INTO "Material" VALUES(2,'Steel Rebar',50,'tons',90,1);
INSERT INTO "Material" VALUES(3,'Asphalt',100,'tons',60,2);
INSERT INTO "Material" VALUES(4,'Gravel',200,'tons',40,2);
CREATE TABLE Project (
    projectID INTEGER PRIMARY KEY,
    name TEXT,
    startDate DATE,
    endDate DATE,
    structuralInspectionID INTEGER,
    safetyInspectionID INTEGER,
    FOREIGN KEY(structuralInspectionID) REFERENCES Structural_Inspection(inspectionID),
    FOREIGN KEY(safetyInspectionID) REFERENCES Safety_Inspection(inspectionID)
);
INSERT INTO "Project" VALUES(1,'Bridge Replacement','2023-01-01','2023-12-31',1,2);
INSERT INTO "Project" VALUES(2,'Highway Repair','2023-02-01','2023-12-31',3,4);
CREATE TABLE Safety_Inspection (
    inspectionID INTEGER PRIMARY KEY,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result TEXT,
    inspectorEmployeeID INTEGER,
    FOREIGN KEY(inspectorEmployeeID) REFERENCES Employee(employeeID)
);
INSERT INTO "Safety_Inspection" VALUES(1,'2023-01-15 10:00:00','Passed',1);
INSERT INTO "Safety_Inspection" VALUES(2,'2023-02-15 10:00:00','Failed',2);
CREATE TABLE Structural_Inspection (
    inspectionID INTEGER PRIMARY KEY,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result TEXT,
    inspectorEmployeeID INTEGER,
    FOREIGN KEY(inspectorEmployeeID) REFERENCES Employee(employeeID)
);
INSERT INTO "Structural_Inspection" VALUES(1,'2023-01-20 10:00:00','Passed',1);
INSERT INTO "Structural_Inspection" VALUES(2,'2023-02-20 10:00:00','Failed',2);
CREATE TABLE Worker (
    workerID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    employmentStartDate DATE,
    currentProjectID INTEGER,
    FOREIGN KEY(currentProjectID) REFERENCES Project(projectID)
);
INSERT INTO "Worker" VALUES(1,'Tom','Hanks','2022-01-01',1);
INSERT INTO "Worker" VALUES(2,'Sam','Waterson','2022-01-01',2);
COMMIT;
