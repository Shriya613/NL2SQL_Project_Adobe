BEGIN TRANSACTION;
CREATE TABLE Assignments(
    ScriptwriterID INTEGER,
    EditorID INTEGER,
    ProjectID INTEGER,
    FOREIGN KEY(ScriptwriterID) REFERENCES Scriptwriters(ID),
    FOREIGN KEY(EditorID) REFERENCES Editors(ID),
    FOREIGN KEY(ProjectID) REFERENCES Projects(ID),
    PRIMARY KEY(ScriptwriterID, EditorID, ProjectID)
);
INSERT INTO "Assignments" VALUES(1,1,1);
INSERT INTO "Assignments" VALUES(1,2,2);
INSERT INTO "Assignments" VALUES(2,1,3);
INSERT INTO "Assignments" VALUES(3,2,4);
INSERT INTO "Assignments" VALUES(1,1,5);
CREATE TABLE Editors(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    HireDate DATE
);
INSERT INTO "Editors" VALUES(1,'Alice Brown','2015-02-10');
INSERT INTO "Editors" VALUES(2,'Bob White','2018-08-20');
INSERT INTO "Editors" VALUES(3,'Carol Green','2009-11-15');
CREATE TABLE Projects(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Deadline DATE,
    Budget REAL 
);
INSERT INTO "Projects" VALUES(1,'Film A','2021-04-21',500000.0);
INSERT INTO "Projects" VALUES(2,'Film B','2021-05-15',750000.0);
INSERT INTO "Projects" VALUES(3,'Film C','2021-04-22',1000000.0);
INSERT INTO "Projects" VALUES(4,'Film D','2021-06-01',1500000.0);
INSERT INTO "Projects" VALUES(5,'Film E','2021-05-10',2000000.0);
CREATE TABLE Scriptwriters(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Birthdate DATE
);
INSERT INTO "Scriptwriters" VALUES(1,'John Doe','1980-06-07');
INSERT INTO "Scriptwriters" VALUES(2,'Jane Smith','1992-09-11');
INSERT INTO "Scriptwriters" VALUES(3,'Michael Johnson','1975-03-14');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Scriptwriters',3);
INSERT INTO "sqlite_sequence" VALUES('Editors',3);
INSERT INTO "sqlite_sequence" VALUES('Projects',5);
COMMIT;
