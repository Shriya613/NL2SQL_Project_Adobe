BEGIN TRANSACTION;
CREATE TABLE Company(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Company" VALUES(1,'yourcompany1');
INSERT INTO "Company" VALUES(2,'yourcompany2');
CREATE TABLE Employee(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES Company(id),
    employee_name TEXT UNIQUE);
INSERT INTO "Employee" VALUES(1,1,'Employee of yourcompany1');
INSERT INTO "Employee" VALUES(2,1,'Another employee of yourcompany1');
INSERT INTO "Employee" VALUES(3,2,'Employee of yourcompany2');
CREATE TABLE Position(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_title TEXT,
    trades_jobs TEXT ,
    company_id INTEGER REFERENCES Company(id));
INSERT INTO "Position" VALUES(1,'Position1','Construction worker',1);
INSERT INTO "Position" VALUES(2,'Position2','Electrician',1);
INSERT INTO "Position" VALUES(3,'Position3','Construction worker',2);
INSERT INTO "Position" VALUES(4,'Position4','Electrician',2);
CREATE TABLE Skill(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_type TEXT,
    position_id INTEGER REFERENCES Position(id));
INSERT INTO "Skill" VALUES(1,'Skill1','Position1');
INSERT INTO "Skill" VALUES(2,'Skill2','Position2');
INSERT INTO "Skill" VALUES(3,'Skill3','Position3');
INSERT INTO "Skill" VALUES(4,'Skill4','Position4');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Company',2);
INSERT INTO "sqlite_sequence" VALUES('Employee',3);
INSERT INTO "sqlite_sequence" VALUES('Position',4);
INSERT INTO "sqlite_sequence" VALUES('Skill',4);
COMMIT;
