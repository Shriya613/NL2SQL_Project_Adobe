BEGIN TRANSACTION;
CREATE TABLE Architects(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INT
);
INSERT INTO "Architects" VALUES(1,'John Doe',10);
INSERT INTO "Architects" VALUES(2,'Jane Smith',7);
INSERT INTO "Architects" VALUES(3,'Michael Johnson',5);
INSERT INTO "Architects" VALUES(4,'Emily Williams',8);
INSERT INTO "Architects" VALUES(5,'David Brown',6);
CREATE TABLE Architects_Building_Types(
    architect_id INTEGER,
    building_type_id INTEGER,
    FOREIGN KEY (architect_id) REFERENCES Architects(id),
    FOREIGN KEY (building_type_id) REFERENCES Building_Types(id),
    PRIMARY KEY (architect_id, building_type_id)
);
INSERT INTO "Architects_Building_Types" VALUES(1,1);
INSERT INTO "Architects_Building_Types" VALUES(1,2);
INSERT INTO "Architects_Building_Types" VALUES(2,1);
INSERT INTO "Architects_Building_Types" VALUES(2,2);
INSERT INTO "Architects_Building_Types" VALUES(3,1);
INSERT INTO "Architects_Building_Types" VALUES(3,2);
INSERT INTO "Architects_Building_Types" VALUES(4,1);
INSERT INTO "Architects_Building_Types" VALUES(4,2);
INSERT INTO "Architects_Building_Types" VALUES(5,1);
INSERT INTO "Architects_Building_Types" VALUES(5,2);
CREATE TABLE Building_Types(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT
);
INSERT INTO "Building_Types" VALUES(1,'Residential');
INSERT INTO "Building_Types" VALUES(2,'Commercial');
INSERT INTO "Building_Types" VALUES(3,'Industrial');
INSERT INTO "Building_Types" VALUES(4,'Institutional');
INSERT INTO "Building_Types" VALUES(5,'Mixed-Use');
CREATE TABLE Job_Postings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    required_skills TEXT,
    salary REAL,
    post_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "Job_Postings" VALUES(1,'Senior Residential Architect','Design residential buildings.','AutoCAD, Revit, SketchUp, knowledge of local building codes',90000.0,'2024-01-30 19:16:14');
INSERT INTO "Job_Postings" VALUES(2,'Junior Commercial Architect','Assist in designing commercial projects.','AutoCAD, Revit, basic understanding of structural systems',60000.0,'2024-01-30 19:16:14');
INSERT INTO "Job_Postings" VALUES(3,'Architectural Designer','Support architects on various project types.','SketchUp, InDesign, Adobe Creative Suite',50000.0,'2024-01-30 19:16:14');
INSERT INTO "Job_Postings" VALUES(4,'Project Manager','Oversee construction projects from start to finish.','PMP certification, strong communication skills, knowledge of contract law',80000.0,'2024-01-30 19:16:14');
INSERT INTO "Job_Postings" VALUES(5,'BIM Coordinator','Create and manage BIM models for projects.','Revit, Navisworks, Dynamo, Rhino',65000.0,'2024-01-30 19:16:14');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Architects',5);
INSERT INTO "sqlite_sequence" VALUES('Building_Types',5);
INSERT INTO "sqlite_sequence" VALUES('Job_Postings',5);
COMMIT;
