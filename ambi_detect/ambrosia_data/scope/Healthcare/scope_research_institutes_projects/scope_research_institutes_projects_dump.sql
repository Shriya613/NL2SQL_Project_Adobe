BEGIN TRANSACTION;
CREATE TABLE Projects(
    ProjectID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT);
INSERT INTO "Projects" VALUES(1,'Cancer_Research');
INSERT INTO "Projects" VALUES(2,'Cardiovascular_Disease_Research');
INSERT INTO "Projects" VALUES(3,'Mental_Health_Research');
INSERT INTO "Projects" VALUES(4,'Infectious_Diseases_Research');
INSERT INTO "Projects" VALUES(5,'Diabetes_Research');
CREATE TABLE Research_Institutes(
    InstituteID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Research_Institutes" VALUES(1,'Institute1','New York');
INSERT INTO "Research_Institutes" VALUES(2,'Institute2','Los Angeles');
INSERT INTO "Research_Institutes" VALUES(3,'Institute3','Chicago');
INSERT INTO "Research_Institutes" VALUES(4,'Institute4','Houston');
INSERT INTO "Research_Institutes" VALUES(5,'Institute5','Philadelphia');
CREATE TABLE Research_Institutes_Projects(
    InstituteID INTEGER,
    ProjectID INTEGER,
    FOREIGN KEY (InstituteID) REFERENCES Research_Institutes(InstituteID),
    FOREIGN KEY (ProjectID) REFERENCES Projects(ProjectID));
INSERT INTO "Research_Institutes_Projects" VALUES(1,1);
INSERT INTO "Research_Institutes_Projects" VALUES(1,2);
INSERT INTO "Research_Institutes_Projects" VALUES(2,1);
INSERT INTO "Research_Institutes_Projects" VALUES(2,3);
INSERT INTO "Research_Institutes_Projects" VALUES(3,2);
INSERT INTO "Research_Institutes_Projects" VALUES(3,4);
INSERT INTO "Research_Institutes_Projects" VALUES(4,1);
INSERT INTO "Research_Institutes_Projects" VALUES(4,5);
INSERT INTO "Research_Institutes_Projects" VALUES(5,3);
INSERT INTO "Research_Institutes_Projects" VALUES(5,4);
INSERT INTO "Research_Institutes_Projects" VALUES(3,1);
INSERT INTO "Research_Institutes_Projects" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Research_Institutes',5);
INSERT INTO "sqlite_sequence" VALUES('Projects',5);
COMMIT;
