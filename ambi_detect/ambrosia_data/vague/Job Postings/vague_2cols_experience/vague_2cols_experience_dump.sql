BEGIN TRANSACTION;
CREATE TABLE Companies (
    CompanyID INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Companies" VALUES(1,'TechCorp','New York');
INSERT INTO "Companies" VALUES(2,'SoftwareInc','San Francisco');
INSERT INTO "Companies" VALUES(3,'AppDevelopers','Boston');
INSERT INTO "Companies" VALUES(4,'CodingWorld','Chicago');
INSERT INTO "Companies" VALUES(5,'ProgrammingCo','Los Angeles');
CREATE TABLE EducationRequirements (
    RequirementID INTEGER PRIMARY KEY,
    DegreeLevel TEXT,
    FieldOfStudy TEXT
);
INSERT INTO "EducationRequirements" VALUES(1,'Masters','Computer Science');
INSERT INTO "EducationRequirements" VALUES(2,'Bachelors','Information Technology');
INSERT INTO "EducationRequirements" VALUES(3,'PhD','Artificial Intelligence');
INSERT INTO "EducationRequirements" VALUES(4,'Associates','Web Development');
INSERT INTO "EducationRequirements" VALUES(5,'Certified','Project Management');
CREATE TABLE Jobs (
    JobID INTEGER PRIMARY KEY,
    Minimum_Years INTEGER ,
    Preferred_Years INTEGER ,
    JobTitle TEXT,
    Description TEXT,
    Salary REAL 
);
INSERT INTO "Jobs" VALUES(1,2,4,'Software Developer','We are looking for a Software Developer with at least 2 years of experience.',70000.0);
INSERT INTO "Jobs" VALUES(2,3,6,'Data Analyst','Experienced Data Analyst required to join our team.',80000.0);
INSERT INTO "Jobs" VALUES(3,1,3,'Graphic Designer','We need an entry-level Graphic Designer to work on various projects.',50000.0);
INSERT INTO "Jobs" VALUES(4,4,8,'Product Manager','An experienced Product Manager is needed to manage and lead product development.',90000.0);
INSERT INTO "Jobs" VALUES(5,5,10,'System Administrator','A System Administrator with extensive experience is required to maintain and improve system performance.',60000.0);
CREATE TABLE RequiredSkills (
    SkillID INTEGER PRIMARY KEY,
    SkillName TEXT,
    ProficiencyLevel TEXT 
);
INSERT INTO "RequiredSkills" VALUES(1,'Python Programming','Advanced');
INSERT INTO "RequiredSkills" VALUES(2,'JavaScript','Intermediate');
INSERT INTO "RequiredSkills" VALUES(3,'SQL','Expert');
INSERT INTO "RequiredSkills" VALUES(4,'HTML/CSS','Beginner');
INSERT INTO "RequiredSkills" VALUES(5,'Agile Methodology','Proficient');
COMMIT;
