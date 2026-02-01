BEGIN TRANSACTION;
CREATE TABLE Departments(
    department_id INTEGER PRIMARY KEY AUTOINCREMENT,
    department_name TEXT
);
INSERT INTO "Departments" VALUES(1,'Foreign_Correspondence');
INSERT INTO "Departments" VALUES(2,'Politics');
INSERT INTO "Departments" VALUES(3,'Economy');
INSERT INTO "Departments" VALUES(4,'Technology');
INSERT INTO "Departments" VALUES(5,'Culture');
CREATE TABLE News_Agencies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    established_year INT
);
INSERT INTO "News_Agencies" VALUES(1,'Agency1','USA',2008);
INSERT INTO "News_Agencies" VALUES(2,'Agency2','UK',1996);
INSERT INTO "News_Agencies" VALUES(3,'Agency3','India',2010);
INSERT INTO "News_Agencies" VALUES(4,'Agency4','Australia',2000);
INSERT INTO "News_Agencies" VALUES(5,'Agency5','Germany',2007);
CREATE TABLE News_Agencies_Departments(
    agency_id INTEGER,
    department_id INTEGER,
    FOREIGN KEY(agency_id) REFERENCES News_Agencies(id),
    FOREIGN KEY(department_id) REFERENCES Departments(department_id),
    PRIMARY KEY(agency_id, department_id)
);
INSERT INTO "News_Agencies_Departments" VALUES(1,1);
INSERT INTO "News_Agencies_Departments" VALUES(1,2);
INSERT INTO "News_Agencies_Departments" VALUES(2,3);
INSERT INTO "News_Agencies_Departments" VALUES(2,4);
INSERT INTO "News_Agencies_Departments" VALUES(3,5);
INSERT INTO "News_Agencies_Departments" VALUES(3,2);
INSERT INTO "News_Agencies_Departments" VALUES(4,3);
INSERT INTO "News_Agencies_Departments" VALUES(4,4);
INSERT INTO "News_Agencies_Departments" VALUES(5,1);
INSERT INTO "News_Agencies_Departments" VALUES(5,2);
INSERT INTO "News_Agencies_Departments" VALUES(2,1);
INSERT INTO "News_Agencies_Departments" VALUES(3,1);
INSERT INTO "News_Agencies_Departments" VALUES(4,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('News_Agencies',5);
INSERT INTO "sqlite_sequence" VALUES('Departments',5);
COMMIT;
