BEGIN TRANSACTION;
CREATE TABLE Graphic_Designers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    experience INTEGER,
    skills TEXT,
    office TEXT
);
INSERT INTO "Graphic_Designers" VALUES(1,'Jane Doe',3,'Adobe Illustrator, Photoshop, InDesign','2/7 London Street');
INSERT INTO "Graphic_Designers" VALUES(2,'John Smith',6,'Illustrator, Sketch, Figma','10 Downing St');
CREATE TABLE HR_Managers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    years_experience INTEGER,
    department TEXT,
    office TEXT
);
INSERT INTO "HR_Managers" VALUES(1,'Mike Davis',8,'HR','2/7 London Street');
INSERT INTO "HR_Managers" VALUES(2,'Emily Wilson',5,'Recruiting','10 Downing St');
CREATE TABLE Recruitment_Coordinators (
    id INTEGER PRIMARY KEY,
    name TEXT,
    job_type TEXT,
    responsibilities TEXT,
    office TEXT
);
INSERT INTO "Recruitment_Coordinators" VALUES(1,'Linda Taylor','Full Time','Sourcing candidates, scheduling interviews','2/7 London Street');
INSERT INTO "Recruitment_Coordinators" VALUES(2,'David Miller','Contract','Vendor management, candidate assessment','10 Downing St');
CREATE TABLE Software_Engineers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    programming_skills TEXT,
    project_history TEXT,
    office TEXT
);
INSERT INTO "Software_Engineers" VALUES(1,'Rachel Green','C#, .NET Core','Bank Management System','2/7 London Street');
INSERT INTO "Software_Engineers" VALUES(2,'Chris Evans','Java, Spring Boot','Inventory Management App','10 Downing St');
CREATE TABLE Web_Developers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    programming_languages TEXT,
    projects TEXT,
    office TEXT
);
INSERT INTO "Web_Developers" VALUES(1,'Alice Johnson','JavaScript, React, Node.js','Ecommerce Website','2/7 London Street');
INSERT INTO "Web_Developers" VALUES(2,'Bob Brown','Python, Django, Flask','Analytics Platform','10 Downing St');
COMMIT;
