BEGIN TRANSACTION;
CREATE TABLE Architect (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Architect" VALUES(1,'John Smith','johnsmith@example.com');
INSERT INTO "Architect" VALUES(2,'Jane Doe','janedoe@example.com');
INSERT INTO "Architect" VALUES(3,'Mike Johnson','mikejohnson@example.com');
INSERT INTO "Architect" VALUES(4,'Emily Davis','emilydavis@example.com');
INSERT INTO "Architect" VALUES(5,'David Brown','davidbrown@example.com');
CREATE TABLE Building (
    id INTEGER PRIMARY KEY,
    building_name TEXT,
    firm_id INTEGER,
    architect_id INTEGER,
    FOREIGN KEY(firm_id) REFERENCES Firm(id),
    FOREIGN KEY(architect_id) REFERENCES Architect(id)
);
INSERT INTO "Building" VALUES(1,'Highrise Tower',1,1);
INSERT INTO "Building" VALUES(2,'Residential Complex',2,2);
INSERT INTO "Building" VALUES(3,'Office Building',3,3);
INSERT INTO "Building" VALUES(4,'School Building',1,4);
INSERT INTO "Building" VALUES(5,'Hospital',2,5);
CREATE TABLE Employee (
    id INTEGER PRIMARY KEY,
    employee_name TEXT,
    position TEXT,
    firm_id INTEGER,
    FOREIGN KEY(firm_id) REFERENCES Firm(id)
);
INSERT INTO "Employee" VALUES(1,'Tom Williams','Project Manager',1);
INSERT INTO "Employee" VALUES(2,'Susan Thompson','Architect',2);
INSERT INTO "Employee" VALUES(3,'Chris Jones','Superintendent',3);
INSERT INTO "Employee" VALUES(4,'Lisa Brown','Engineer',1);
INSERT INTO "Employee" VALUES(5,'Mark Taylor','Electrician',2);
CREATE TABLE Firm (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT
);
INSERT INTO "Firm" VALUES(1,'ABC Construction','123 Main St, Anytown, USA');
INSERT INTO "Firm" VALUES(2,'XYZ Builders','456 Elm St, Anycity, USA');
INSERT INTO "Firm" VALUES(3,'PQR Contractors','789 Oak St, Anothertown, USA');
CREATE TABLE Material (
    id INTEGER PRIMARY KEY,
    material_type TEXT,
    cost REAL
);
INSERT INTO "Material" VALUES(1,'Concrete',100.0);
INSERT INTO "Material" VALUES(2,'Steel',200.0);
INSERT INTO "Material" VALUES(3,'Glass',150.0);
INSERT INTO "Material" VALUES(4,'Wood',50.0);
INSERT INTO "Material" VALUES(5,'Bricks',75.0);
CREATE TABLE Project (
    id INTEGER PRIMARY KEY,
    project_name TEXT,
    building_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY(building_id) REFERENCES Building(id)
);
INSERT INTO "Project" VALUES(1,'Tower Foundation',1,'2022-01-01','2022-03-31');
INSERT INTO "Project" VALUES(2,'Complex Framework',2,'2022-02-01','2022-04-30');
INSERT INTO "Project" VALUES(3,'Office Interior',3,'2022-03-01','2022-05-31');
INSERT INTO "Project" VALUES(4,'School Renovation',1,'2022-04-01','2022-06-30');
INSERT INTO "Project" VALUES(5,'Hospital Expansion',2,'2022-05-01','2022-07-31');
COMMIT;
