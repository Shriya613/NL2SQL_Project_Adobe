BEGIN TRANSACTION;
CREATE TABLE Architect_Projects (
    architect_id INTEGER,
    project_id INTEGER,
    FOREIGN KEY (architect_id) REFERENCES Architects(id),
    FOREIGN KEY (project_id) REFERENCES Projects(project_id),
    PRIMARY KEY (architect_id, project_id)
);
INSERT INTO "Architect_Projects" VALUES(1,1);
INSERT INTO "Architect_Projects" VALUES(2,3);
INSERT INTO "Architect_Projects" VALUES(3,5);
INSERT INTO "Architect_Projects" VALUES(4,2);
INSERT INTO "Architect_Projects" VALUES(5,4);
CREATE TABLE Architects (
    id INTEGER PRIMARY KEY,
    name TEXT,
    years_in_practice INT ,
    design_software_expertise TEXT,
    salary REAL
);
INSERT INTO "Architects" VALUES(1,'Lisa Miller',8,'AutoCAD',85000.0);
INSERT INTO "Architects" VALUES(2,'William Taylor',5,'Revit',70000.0);
INSERT INTO "Architects" VALUES(3,'Olivia Davis',7,'AutoCAD',80000.0);
INSERT INTO "Architects" VALUES(4,'Samuel Moore',9,'SketchUp',95000.0);
INSERT INTO "Architects" VALUES(5,'Ethan Scott',3,'Rhino',65000.0);
CREATE TABLE Civil_Engineers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    years_of_experience INT ,
    design_software_expertise TEXT,
    salary REAL
);
INSERT INTO "Civil_Engineers" VALUES(1,'John Doe',7,'AutoCAD',80000.0);
INSERT INTO "Civil_Engineers" VALUES(2,'Jane Smith',3,'Revit',65000.0);
INSERT INTO "Civil_Engineers" VALUES(3,'Michael Johnson',5,'AutoCAD',75000.0);
INSERT INTO "Civil_Engineers" VALUES(4,'Emily Brown',9,'SketchUp',90000.0);
INSERT INTO "Civil_Engineers" VALUES(5,'David Wilson',2,'Rhino',60000.0);
CREATE TABLE Design_Softwares (
    software_name TEXT PRIMARY KEY,
    popularity INT 
);
INSERT INTO "Design_Softwares" VALUES('AutoCAD',90);
INSERT INTO "Design_Softwares" VALUES('Revit',80);
INSERT INTO "Design_Softwares" VALUES('SketchUp',70);
INSERT INTO "Design_Softwares" VALUES('Rhino',50);
INSERT INTO "Design_Softwares" VALUES('ArchiCAD',60);
CREATE TABLE Engineer_Projects (
    engineer_id INTEGER,
    project_id INTEGER,
    FOREIGN KEY (engineer_id) REFERENCES Civil_Engineers(id),
    FOREIGN KEY (project_id) REFERENCES Projects(project_id),
    PRIMARY KEY (engineer_id, project_id)
);
INSERT INTO "Engineer_Projects" VALUES(1,1);
INSERT INTO "Engineer_Projects" VALUES(1,2);
INSERT INTO "Engineer_Projects" VALUES(2,3);
INSERT INTO "Engineer_Projects" VALUES(3,4);
INSERT INTO "Engineer_Projects" VALUES(4,5);
CREATE TABLE Projects (
    project_id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    start_date DATE,
    end_date DATE
);
INSERT INTO "Projects" VALUES(1,'Bridge Construction','Building a new bridge across the city river.','2021-05-01','2022-07-31');
INSERT INTO "Projects" VALUES(2,'Office Building','Constructing a new office building downtown.','2021-06-01','2022-10-31');
INSERT INTO "Projects" VALUES(3,'Residential Complex','Developing a residential complex on the outskirts of town.','2021-07-01','2023-07-31');
INSERT INTO "Projects" VALUES(4,'Highway Expansion','Expanding a busy highway to reduce traffic congestion.','2021-08-01','2023-12-31');
INSERT INTO "Projects" VALUES(5,'Shopping Mall','Building a shopping mall with multiple stores and entertainment options.','2021-09-01','2024-01-31');
COMMIT;
