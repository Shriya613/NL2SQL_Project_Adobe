BEGIN TRANSACTION;
CREATE TABLE Accountants(
    id INTEGER PRIMARY KEY,
    name TEXT,
    company_name TEXT,
    education TEXT,
    experience_years INT);
INSERT INTO "Accountants" VALUES(1,'Jennifer Lawrence','FinanceGroup','BBA',6);
INSERT INTO "Accountants" VALUES(2,'Scarlett Johansson','AccountingFirm','MAcc',7);
CREATE TABLE Construction_Workers(
    id INTEGER PRIMARY KEY,
    name TEXT,
    company_owner TEXT,
    work_experience INT,
    salary REAL);
INSERT INTO "Construction_Workers" VALUES(1,'John Doe','Derek Brown',7,30000.0);
INSERT INTO "Construction_Workers" VALUES(2,'Jane Smith','Michael Johnson',5,28000.0);
INSERT INTO "Construction_Workers" VALUES(3,'Robert White','Emily Davis',6,29000.0);
CREATE TABLE Electricians(
    id INTEGER PRIMARY KEY,
    name TEXT,
    company_owner TEXT,
    certifications TEXT,
    hourly_rate REAL);
INSERT INTO "Electricians" VALUES(1,'Tom Anderson','Derek Brown','A, B, C',25.0);
INSERT INTO "Electricians" VALUES(2,'Erica Thompson','Samantha Taylor','X, Y, Z',27.0);
CREATE TABLE HR_Specialists(
    id INTEGER PRIMARY KEY,
    name TEXT,
    company_name TEXT,
    industry TEXT,
    years_in_industry INT);
INSERT INTO "HR_Specialists" VALUES(1,'Chris Hemsworth','HumanResources','HR',8);
INSERT INTO "HR_Specialists" VALUES(2,'Mark Ruffalo','RecruitmentAgencies','IT',10);
CREATE TABLE Marketing_Managers(
    id INTEGER PRIMARY KEY,
    name TEXT,
    company_name TEXT,
    marketing_strategy TEXT,
    campaign_successes TEXT);
INSERT INTO "Marketing_Managers" VALUES(1,'Ryan Reynolds','AdvertisingAgency','Content Marketing','Product Launch A');
INSERT INTO "Marketing_Managers" VALUES(2,'Ben Affleck','BrandConsultancy','SEO','Website Traffic Increase');
CREATE TABLE Software_Engineers(
    id INTEGER PRIMARY KEY,
    name TEXT,
    company_name TEXT,
    programming_language TEXT,
    years_of_experience INT);
INSERT INTO "Software_Engineers" VALUES(1,'Chris Evans','TechCorp','Python',4);
INSERT INTO "Software_Engineers" VALUES(2,'Olivia Wilde','SoftwareInc','JavaScript',3);
INSERT INTO "Software_Engineers" VALUES(3,'Gal Gadot','CodingCo','C++',5);
COMMIT;
