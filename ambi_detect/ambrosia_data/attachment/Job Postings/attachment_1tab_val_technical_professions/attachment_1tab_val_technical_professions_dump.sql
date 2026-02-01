BEGIN TRANSACTION;
CREATE TABLE Companies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    email TEXT);
INSERT INTO "Companies" VALUES(1,'Company A','123 Main St, Anytown, USA','info@companya.com');
INSERT INTO "Companies" VALUES(2,'Company B','456 Elm St, Othertown, USA','contact@companyb.com');
CREATE TABLE Jobs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    companyId INTEGER,
    salary INT,
    technicalProfession TEXT,
    skillIds TEXT,
    locationId INTEGER,
    FOREIGN KEY(companyId) REFERENCES Companies(id),
    FOREIGN KEY(skillIds) REFERENCES Technologies(id));
INSERT INTO "Jobs" VALUES(1,'Software Engineer','Develop software applications using Python.',1,80000,'Engineer','2,3',1);
INSERT INTO "Jobs" VALUES(2,'Senior Software Engineer','Lead a team of engineers and develop complex software systems.',1,95000,'Engineer','2,3',1);
INSERT INTO "Jobs" VALUES(3,'Software Architect','Design and implement high-level architecture for large-scale applications.',2,80000,'Architect','1,4',2);
INSERT INTO "Jobs" VALUES(4,'Principal Software Architect','Define product vision and guide the overall technical direction of the project.',2,105000,'Architect','1,4',2);
CREATE TABLE Locations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    country TEXT);
INSERT INTO "Locations" VALUES(1,'Anytown','USA');
INSERT INTO "Locations" VALUES(2,'Othertown','USA');
CREATE TABLE Technologies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    technologyName TEXT UNIQUE);
INSERT INTO "Technologies" VALUES(1,'Python');
INSERT INTO "Technologies" VALUES(2,'JavaScript');
INSERT INTO "Technologies" VALUES(3,'C#');
INSERT INTO "Technologies" VALUES(4,'Ruby');
INSERT INTO "Technologies" VALUES(5,'Swift');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',2);
INSERT INTO "sqlite_sequence" VALUES('Locations',2);
INSERT INTO "sqlite_sequence" VALUES('Technologies',5);
INSERT INTO "sqlite_sequence" VALUES('Jobs',4);
COMMIT;
