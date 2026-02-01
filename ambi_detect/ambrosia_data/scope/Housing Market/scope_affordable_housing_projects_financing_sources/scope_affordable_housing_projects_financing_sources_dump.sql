BEGIN TRANSACTION;
CREATE TABLE Affordable_Housing_Projects(
    project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT,
    location TEXT);
INSERT INTO "Affordable_Housing_Projects" VALUES(1,'Project1','Location1');
INSERT INTO "Affordable_Housing_Projects" VALUES(2,'Project2','Location2');
INSERT INTO "Affordable_Housing_Projects" VALUES(3,'Project3','Location3');
INSERT INTO "Affordable_Housing_Projects" VALUES(4,'Project4','Location4');
INSERT INTO "Affordable_Housing_Projects" VALUES(5,'Project5','Location5');
CREATE TABLE Affordable_Housing_Projects_Financing_Sources(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    housing_project_id INTEGER,
    financing_source_id INTEGER,
    FOREIGN KEY(housing_project_id) REFERENCES Affordable_Housing_Projects(project_id),
    FOREIGN KEY(financing_source_id) REFERENCES Financing_Sources(source_id));
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(1,1,1);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(2,1,2);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(3,2,1);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(4,2,2);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(5,3,1);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(6,3,3);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(7,4,1);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(8,4,4);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(9,5,1);
INSERT INTO "Affordable_Housing_Projects_Financing_Sources" VALUES(10,5,5);
CREATE TABLE Financing_Sources(
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_value TEXT);
INSERT INTO "Financing_Sources" VALUES(1,'Low_Income_Housing_Tax_Credit');
INSERT INTO "Financing_Sources" VALUES(2,'Government_Grants');
INSERT INTO "Financing_Sources" VALUES(3,'Private_Investments');
INSERT INTO "Financing_Sources" VALUES(4,'Public_Private_Partnerships');
INSERT INTO "Financing_Sources" VALUES(5,'Bonds');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Affordable_Housing_Projects',5);
INSERT INTO "sqlite_sequence" VALUES('Financing_Sources',5);
INSERT INTO "sqlite_sequence" VALUES('Affordable_Housing_Projects_Financing_Sources',10);
COMMIT;
