BEGIN TRANSACTION;
CREATE TABLE EmploymentStats (
    Year INTEGER PRIMARY KEY,
    TotalNumberOfJobs INTEGER,
    UnemployedCount INTEGER,
    UnemploymentRate REAL,
    MinimumWage REAL,
    MaximumWage REAL
);
INSERT INTO "EmploymentStats" VALUES(2020,150000000,10000000,6.67,7.25,75000.0);
CREATE TABLE HousingData (
    HouseID INTEGER PRIMARY KEY,
    Address TEXT,
    Type TEXT,
    Price REAL,
    SizeInSquareFeet INTEGER,
    Bedrooms INTEGER,
    Bathrooms INTEGER,
    Age INTEGER,
    Neighborhood TEXT
);
INSERT INTO "HousingData" VALUES(1,'123 Main St','Condo',250000.0,1000,2,1,5,'Uptown');
INSERT INTO "HousingData" VALUES(2,'456 Elm St','Single Family Home',500000.0,2500,4,2,10,'Suburbia');
CREATE TABLE Industries (
    IndustryID INTEGER PRIMARY KEY,
    Primary_Industry TEXT,
    Most_Common_Job TEXT,
    Region TEXT,
    NumberOfEmployees INTEGER,
    AnnualRevenue REAL
);
INSERT INTO "Industries" VALUES(1,'Construction','Laborer','Northeast',2000,500000.0);
INSERT INTO "Industries" VALUES(2,'Technology','Software Developer','Southwest',3000,600000.0);
CREATE TABLE Jobs (
    JobID INTEGER PRIMARY KEY,
    Title TEXT,
    Description TEXT,
    SkillsRequired TEXT,
    AverageSalary REAL,
    GrowthProspects TEXT,
    RequiredEducation TEXT
);
INSERT INTO "Jobs" VALUES(1,'Software Engineer','Design, develop, and maintain software applications.','Programming languages, problem solving skills, teamwork',80000.0,'High','Bachelor''s degree in Computer Science');
INSERT INTO "Jobs" VALUES(2,'Registered Nurse','Provide and coordinate patient care, educate patients and the public about various aspects of health','Communication skills, critical thinking, empathy',70000.0,'Good','Associate degree in Nursing');
CREATE TABLE People (
    PersonID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Gender CHAR(1) ,
    Race TEXT,
    Ethnicity TEXT,
    MaritalStatus TEXT,
    EducationLevel TEXT,
    Occupation TEXT
);
INSERT INTO "People" VALUES(1,'John','Doe','1985-07-15','M','White','Hispanic','Single','Bachelor''s Degree','Engineer');
INSERT INTO "People" VALUES(2,'Jane','Smith','1990-02-20','F','Black','Non-Hispanic','Married','Master''s Degree','Teacher');
CREATE TABLE Regions (
    RegionID INTEGER PRIMARY KEY,
    Name TEXT,
    Population INTEGER,
    AreaInSqKm REAL,
    CapitalCity TEXT,
    Climate TEXT,
    MajorCities TEXT
);
INSERT INTO "Regions" VALUES(1,'New England',16000000,89000.0,'Boston','Continental climate','Boston, Providence, Hartford');
INSERT INTO "Regions" VALUES(2,'Great Lakes',52000000,250000.0,'Chicago','Humid continental climate','Chicago, Detroit, Cleveland');
COMMIT;
