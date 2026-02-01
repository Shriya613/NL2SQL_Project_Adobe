BEGIN TRANSACTION;
CREATE TABLE "Building" (
    id INTEGER PRIMARY KEY,
    building_name TEXT,
    architect_id INTEGER,
    FOREIGN KEY(architect_id) REFERENCES "Designer"(id)
);
INSERT INTO "Building" VALUES(1,'Highrise Tower',1);
INSERT INTO "Building" VALUES(2,'Residential Complex',2);
INSERT INTO "Building" VALUES(3,'Office Building',3);
INSERT INTO "Building" VALUES(4,'School Building',4);
INSERT INTO "Building" VALUES(5,'Hospital',5);
CREATE TABLE "Designer" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Designer" VALUES(1,'John Smith','johnsmith@example.com');
INSERT INTO "Designer" VALUES(2,'Jane Doe','janedoe@example.com');
INSERT INTO "Designer" VALUES(3,'Mike Johnson','mikejohnson@example.com');
INSERT INTO "Designer" VALUES(4,'Emily Davis','emilydavis@example.com');
INSERT INTO "Designer" VALUES(5,'David Brown','davidbrown@example.com');
CREATE TABLE "Employee" (
    id INTEGER PRIMARY KEY,
    employee_name TEXT,
    position TEXT);
INSERT INTO "Employee" VALUES(1,'Tom Williams','Project Manager');
INSERT INTO "Employee" VALUES(2,'Susan Thompson','Architect');
INSERT INTO "Employee" VALUES(3,'Chris Jones','Superintendent');
INSERT INTO "Employee" VALUES(4,'Lisa Brown','Engineer');
INSERT INTO "Employee" VALUES(5,'Mark Taylor','Electrician');
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
