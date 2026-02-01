BEGIN TRANSACTION;
CREATE TABLE Methodologies(
    methodology_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Methodologies" VALUES(1,'Agile');
INSERT INTO "Methodologies" VALUES(2,'Waterfall');
INSERT INTO "Methodologies" VALUES(3,'Scrum');
INSERT INTO "Methodologies" VALUES(4,'Kanban');
INSERT INTO "Methodologies" VALUES(5,'Lean');
CREATE TABLE Other_Tables1(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column2 TEXT,
    column3 DATE
);
INSERT INTO "Other_Tables1" VALUES(1,'Data A','2021-01-01');
INSERT INTO "Other_Tables1" VALUES(2,'Data B','2021-02-01');
INSERT INTO "Other_Tables1" VALUES(3,'Data C','2021-03-01');
INSERT INTO "Other_Tables1" VALUES(4,'Data D','2021-04-01');
INSERT INTO "Other_Tables1" VALUES(5,'Data E','2021-05-01');
CREATE TABLE Other_Tables2(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    columnA INT ,
    columnB FLOAT 
);
INSERT INTO "Other_Tables2" VALUES(1,1,2.5);
INSERT INTO "Other_Tables2" VALUES(2,2,3.5);
INSERT INTO "Other_Tables2" VALUES(3,3,4.5);
INSERT INTO "Other_Tables2" VALUES(4,4,5.5);
INSERT INTO "Other_Tables2" VALUES(5,5,6.5);
CREATE TABLE Project_Managers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    years_of_experience INT 
);
INSERT INTO "Project_Managers" VALUES(1,'John Doe',10);
INSERT INTO "Project_Managers" VALUES(2,'Jane Smith',8);
INSERT INTO "Project_Managers" VALUES(3,'Mike Brown',6);
INSERT INTO "Project_Managers" VALUES(4,'Emily Davis',7);
INSERT INTO "Project_Managers" VALUES(5,'David Johnson',9);
CREATE TABLE Project_Managers_Methodologies(
    project_manager_id INTEGER,
    methodology_id INTEGER,
    FOREIGN KEY (project_manager_id) REFERENCES Project_Managers(id),
    FOREIGN KEY (methodology_id) REFERENCES Methodologies(methodology_id)
);
INSERT INTO "Project_Managers_Methodologies" VALUES(1,1);
INSERT INTO "Project_Managers_Methodologies" VALUES(2,1);
INSERT INTO "Project_Managers_Methodologies" VALUES(3,1);
INSERT INTO "Project_Managers_Methodologies" VALUES(4,1);
INSERT INTO "Project_Managers_Methodologies" VALUES(5,1);
INSERT INTO "Project_Managers_Methodologies" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Project_Managers',5);
INSERT INTO "sqlite_sequence" VALUES('Methodologies',5);
INSERT INTO "sqlite_sequence" VALUES('Other_Tables1',5);
INSERT INTO "sqlite_sequence" VALUES('Other_Tables2',5);
COMMIT;
