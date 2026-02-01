BEGIN TRANSACTION;
CREATE TABLE Applications(
    id INTEGER PRIMARY KEY,
    ApplicantID INT,
    JobID INT,
    ApplicationDate DATE,
    Status TEXT
);
INSERT INTO "Applications" VALUES(1,1,1,'2022-01-05','Pending');
INSERT INTO "Applications" VALUES(2,2,2,'2022-01-06','Rejected');
INSERT INTO "Applications" VALUES(3,3,3,'2022-01-07','Accepted');
INSERT INTO "Applications" VALUES(4,4,4,'2022-01-08','Interviewing');
CREATE TABLE Companies(
    id INTEGER PRIMARY KEY,
    CompanyName TEXT,
    Location TEXT,
    Size INT
);
INSERT INTO "Companies" VALUES(1,'OceanTech Inc.','New York',500);
INSERT INTO "Companies" VALUES(2,'NatureDiscoverers','Los Angeles',200);
INSERT INTO "Companies" VALUES(3,'WildlifeGuardians','San Francisco',300);
INSERT INTO "Companies" VALUES(4,'GreenPlanet','Seattle',400);
CREATE TABLE Jobs(
    id INTEGER PRIMARY KEY,
    PositionTitle TEXT,
    Description TEXT,
    Salary FLOAT,
    PostedDate DATE
);
INSERT INTO "Jobs" VALUES(1,'Marine Biologist','Research on coral reef ecosystems',60000.0,'2022-01-01');
INSERT INTO "Jobs" VALUES(2,'Zoologist','Study of primate behavior',55000.0,'2022-01-01');
INSERT INTO "Jobs" VALUES(3,'Software Engineer','Design and develop software applications',70000.0,'2022-01-02');
INSERT INTO "Jobs" VALUES(4,'Data Analyst','Analyze data sets to provide insights',50000.0,'2022-01-02');
CREATE TABLE Marine_Biologists(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Specialization TEXT,
    Experience TEXT
);
INSERT INTO "Marine_Biologists" VALUES(1,'Alice',30,'Coral Reefs','5 years');
INSERT INTO "Marine_Biologists" VALUES(2,'Bob',28,'Whales','2 years');
INSERT INTO "Marine_Biologists" VALUES(3,'Charlie',35,'Fish Behaviour','7 years');
INSERT INTO "Marine_Biologists" VALUES(4,'Diana',29,'Microbial Ecology','4 years');
CREATE TABLE Skills(
    id INTEGER PRIMARY KEY,
    SkillName TEXT,
    Level TEXT ,
    ProficiencyScore INT
);
INSERT INTO "Skills" VALUES(1,'Scuba Diving','Advanced',85);
INSERT INTO "Skills" VALUES(2,'Field Research','Expert',90);
INSERT INTO "Skills" VALUES(3,'Data Analysis','Beginner',45);
INSERT INTO "Skills" VALUES(4,'Programming','Intermediate',70);
INSERT INTO "Skills" VALUES(5,'Public Speaking','Proficient',80);
CREATE TABLE Zoologists(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Specialization TEXT,
    Experience TEXT
);
INSERT INTO "Zoologists" VALUES(1,'Eve',32,'Primates','5 years');
INSERT INTO "Zoologists" VALUES(2,'Frank',26,'Herpetology','3 years');
INSERT INTO "Zoologists" VALUES(3,'Grace',31,'Entomology','8 years');
INSERT INTO "Zoologists" VALUES(4,'Hannah',27,'Ornithology','6 years');
COMMIT;
