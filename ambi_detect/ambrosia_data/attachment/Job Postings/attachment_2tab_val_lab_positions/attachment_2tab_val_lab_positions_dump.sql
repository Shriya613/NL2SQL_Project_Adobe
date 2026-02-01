BEGIN TRANSACTION;
CREATE TABLE Accountants(
    id INTEGER PRIMARY KEY,
    name TEXT,
    qualification TEXT,
    years_in_industry INTEGER,
    gender TEXT 
);
INSERT INTO "Accountants" VALUES(1,'Anna Johnson','CPA',6,'Female');
INSERT INTO "Accountants" VALUES(2,'William Anderson','CA',8,'Male');
CREATE TABLE Data_Analysts(
    id INTEGER PRIMARY KEY,
    name TEXT,
    specialization TEXT,
    education TEXT,
    gender TEXT 
);
INSERT INTO "Data_Analysts" VALUES(1,'Eva Red','Data Science','Master of Computer Applications','Female');
INSERT INTO "Data_Analysts" VALUES(2,'Steve Black','Business Analytics','MBA','Male');
CREATE TABLE HR_Specialists(
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    employment_duration INTEGER,
    gender TEXT 
);
INSERT INTO "HR_Specialists" VALUES(1,'Carol White','HR',7,'Female');
INSERT INTO "HR_Specialists" VALUES(2,'Robert Green','HR',10,'Male');
CREATE TABLE Laboratory_Technicians(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    experience INTEGER,
    Gender TEXT 
);
INSERT INTO "Laboratory_Technicians" VALUES(1,'Emma Brown',27,3,'Female');
INSERT INTO "Laboratory_Technicians" VALUES(2,'Tom Green',32,6,'Male');
CREATE TABLE Research_Scientists(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    salary REAL,
    Gender TEXT 
);
INSERT INTO "Research_Scientists" VALUES(1,'Jane Doe',30,85000.0,'Female');
INSERT INTO "Research_Scientists" VALUES(2,'John Smith',35,90000.0,'Male');
CREATE TABLE Software_Engineers(
    id INTEGER PRIMARY KEY,
    name TEXT,
    skills TEXT,
    years_experience INTEGER,
    gender TEXT 
);
INSERT INTO "Software_Engineers" VALUES(1,'Alice Blue','Python, Java',4,'Female');
INSERT INTO "Software_Engineers" VALUES(2,'Bob White','C++, Python',5,'Male');
COMMIT;
