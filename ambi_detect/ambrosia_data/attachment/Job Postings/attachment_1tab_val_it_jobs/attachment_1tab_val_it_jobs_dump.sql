BEGIN TRANSACTION;
CREATE TABLE CompanyInfo(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    industry TEXT);
INSERT INTO "CompanyInfo" VALUES(1,'TechCorp','New York','Technology');
INSERT INTO "CompanyInfo" VALUES(2,'FinanceInc','Chicago','Banking');
INSERT INTO "CompanyInfo" VALUES(3,'HealthCareSystems','Los Angeles','Healthcare');
INSERT INTO "CompanyInfo" VALUES(4,'EduCo','San Francisco','Education');
INSERT INTO "CompanyInfo" VALUES(5,'GreenEnergy','Houston','Renewable Energy');
CREATE TABLE EducationRequirements(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    degreeLevel TEXT,
    yearsOfExperience INT);
INSERT INTO "EducationRequirements" VALUES(1,'Bachelor''s Degree',2);
INSERT INTO "EducationRequirements" VALUES(2,'Master''s Degree',3);
INSERT INTO "EducationRequirements" VALUES(3,'PhD',5);
INSERT INTO "EducationRequirements" VALUES(4,'Associate Degree',1);
INSERT INTO "EducationRequirements" VALUES(5,'High School Diploma',0);
CREATE TABLE JobLocations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    state TEXT,
    country TEXT);
INSERT INTO "JobLocations" VALUES(1,'New York','NY','USA');
INSERT INTO "JobLocations" VALUES(2,'Chicago','IL','USA');
INSERT INTO "JobLocations" VALUES(3,'Los Angeles','CA','USA');
INSERT INTO "JobLocations" VALUES(4,'San Francisco','CA','USA');
INSERT INTO "JobLocations" VALUES(5,'Houston','TX','USA');
CREATE TABLE JobPostings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    itJobs TEXT, -- This field contains 'Network Administrator', 'IT Consultant', etc.
    salary REAL, -- This can store values like $70,000
    companyId INTEGER,
    locationId INTEGER,
    skillsId INTEGER,
    educationId INTEGER,
    FOREIGN KEY (companyId) REFERENCES CompanyInfo(id),
    FOREIGN KEY (locationId) REFERENCES JobLocations(id),
    FOREIGN KEY (skillsId) REFERENCES SkillsRequired(id),
    FOREIGN KEY (educationId) REFERENCES EducationRequirements(id));
INSERT INTO "JobPostings" VALUES(1,'Network Administrator','Administer network devices and systems.','Network Administrator','$70,000',1,1,1,1);
INSERT INTO "JobPostings" VALUES(2,'Senior Network Administrator','Manage network infrastructure and security.','Network Administrator','$80,000',1,2,1,1);
INSERT INTO "JobPostings" VALUES(3,'IT Consultant','Advise clients on IT strategy and solutions.','IT Consultant','$70,000',2,3,2,2);
INSERT INTO "JobPostings" VALUES(4,'Senior IT Consultant','Implement IT projects and manage change.','IT Consultant','$90,000',2,4,2,2);
CREATE TABLE SkillsRequired(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skillName TEXT);
INSERT INTO "SkillsRequired" VALUES(1,'Python Programming');
INSERT INTO "SkillsRequired" VALUES(2,'Java Development');
INSERT INTO "SkillsRequired" VALUES(3,'SQL Database Management');
INSERT INTO "SkillsRequired" VALUES(4,'Data Analysis');
INSERT INTO "SkillsRequired" VALUES(5,'Project Management');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('CompanyInfo',5);
INSERT INTO "sqlite_sequence" VALUES('JobLocations',5);
INSERT INTO "sqlite_sequence" VALUES('SkillsRequired',5);
INSERT INTO "sqlite_sequence" VALUES('EducationRequirements',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',4);
COMMIT;
