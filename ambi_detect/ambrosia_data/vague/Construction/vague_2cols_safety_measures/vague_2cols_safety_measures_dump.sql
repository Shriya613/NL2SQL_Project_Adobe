BEGIN TRANSACTION;
CREATE TABLE Employee (
    employeeID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    hireDate DATE,
    position TEXT);
INSERT INTO "Employee" VALUES(1,'John','Doe','2021-01-01','Project Manager');
INSERT INTO "Employee" VALUES(2,'Jane','Smith','2021-02-01','Engineer');
INSERT INTO "Employee" VALUES(3,'Mike','Johnson','2021-03-01','Construction Worker');
INSERT INTO "Employee" VALUES(4,'Emily','Brown','2021-04-01','Supervisor');
INSERT INTO "Employee" VALUES(5,'Tom','White','2021-05-01','Architect');
CREATE TABLE Inventory (
    inventoryID INTEGER PRIMARY KEY,
    materialID INTEGER,
    quantity INTEGER DEFAULT 0,
    FOREIGN KEY(materialID) REFERENCES Material(materialID));
INSERT INTO "Inventory" VALUES(1,1,50);
INSERT INTO "Inventory" VALUES(2,2,100);
INSERT INTO "Inventory" VALUES(3,3,200);
INSERT INTO "Inventory" VALUES(4,4,100);
INSERT INTO "Inventory" VALUES(5,5,50);
CREATE TABLE Material (
    materialID INTEGER PRIMARY KEY,
    name TEXT,
    quantity INTEGER DEFAULT 0,
    cost REAL);
INSERT INTO "Material" VALUES(1,'Cement',100,39.99);
INSERT INTO "Material" VALUES(2,'Steel Rebars',200,68.75);
INSERT INTO "Material" VALUES(3,'Concrete Blocks',300,25.0);
INSERT INTO "Material" VALUES(4,'Bricks',150,18.5);
INSERT INTO "Material" VALUES(5,'Electrical Wiring',100,45.0);
CREATE TABLE Project (
    projectID INTEGER PRIMARY KEY,
    projectName TEXT,
    startDate DATE,
    endDate DATE,
    budget REAL);
INSERT INTO "Project" VALUES(1,'Residential Building','2021-06-01','2021-08-01',300000.0);
INSERT INTO "Project" VALUES(2,'Commercial Complex','2021-07-01','2021-09-01',1000000.0);
INSERT INTO "Project" VALUES(3,'Road Repair','2021-08-01','2021-10-01',500000.0);
INSERT INTO "Project" VALUES(4,'School Renovation','2021-09-01','2021-11-01',600000.0);
INSERT INTO "Project" VALUES(5,'Office Building','2021-10-01','2021-12-01',700000.0);
CREATE TABLE Safety (
    safetyID INTEGER PRIMARY KEY,
    employeeID INTEGER,
    personalProtectiveEquipment TEXT,
    trainingPrograms TEXT,
    constructionWorker TEXT,
    FOREIGN KEY(employeeID) REFERENCES Employee(employeeID));
INSERT INTO "Safety" VALUES(1,1,'Hard Hat, Gloves, Steel-toed Boots','OSHA Training, First Aid Certification','Experienced');
INSERT INTO "Safety" VALUES(2,2,'Helmet, Earplugs, Safety Glasses','Fall Protection Training, CPR/First Aid','Intermediate');
INSERT INTO "Safety" VALUES(3,3,'Hard Hat, Gloves, Safety Vest','Toolbox Talks, Scaffolding Training','Novice');
INSERT INTO "Safety" VALUES(4,4,'Helmet, Safety Shoes, Goggles','Hazard Communication Training, Forklift Operation','Skilled');
INSERT INTO "Safety" VALUES(5,5,'Hard Hat, Gloves, Knee Pads','Lockout/Tagout Training, Confined Space Entry','Expert');
COMMIT;
