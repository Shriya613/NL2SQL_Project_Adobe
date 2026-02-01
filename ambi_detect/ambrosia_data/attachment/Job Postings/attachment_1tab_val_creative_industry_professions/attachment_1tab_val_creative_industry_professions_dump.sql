BEGIN TRANSACTION;
CREATE TABLE Company(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    website TEXT UNIQUE);
INSERT INTO "Company" VALUES(1,'Company A','Address A','www.companya.com');
INSERT INTO "Company" VALUES(2,'Company B','Address B','www.companyb.com');
INSERT INTO "Company" VALUES(3,'Company C','Address C','www.companys.com');
INSERT INTO "Company" VALUES(4,'Company D','Address D','www.companyd.com');
INSERT INTO "Company" VALUES(5,'Company E','Address E','www.companye.com');
CREATE TABLE Department(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES Company(id) ON DELETE CASCADE,
    name TEXT UNIQUE);
INSERT INTO "Department" VALUES(1,1,'Department A');
INSERT INTO "Department" VALUES(2,1,'Department B');
INSERT INTO "Department" VALUES(3,2,'Department X');
INSERT INTO "Department" VALUES(4,3,'Department Y');
INSERT INTO "Department" VALUES(5,4,'Department Z');
CREATE TABLE Employee(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER REFERENCES Position(id) ON DELETE CASCADE,
    employee_name TEXT,
    hire_date DATE);
INSERT INTO "Employee" VALUES(1,1,'John Doe','2021-01-01');
INSERT INTO "Employee" VALUES(2,2,'Jane Smith','2021-02-01');
INSERT INTO "Employee" VALUES(3,3,'Alice Johnson','2021-03-01');
INSERT INTO "Employee" VALUES(4,4,'Bob Brown','2021-04-01');
INSERT INTO "Employee" VALUES(5,5,'Charlie Green','2021-05-01');
CREATE TABLE Position(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    department_id INTEGER REFERENCES Department(id) ON DELETE CASCADE,
    title TEXT,
    creative_industry_profession TEXT ,
    office TEXT,
    salary NUMERIC(10,2));
INSERT INTO "Position" VALUES(1,1,'Position A','Graphic Designer','2/7 London Street',60000);
INSERT INTO "Position" VALUES(2,1,'Position B','Software Engineer','2/9 London Street',80000);
INSERT INTO "Position" VALUES(3,2,'Position C','Graphic Designer','3/4 New York Avenue',55000);
INSERT INTO "Position" VALUES(4,3,'Position D','Data Analyst','3/2 San Francisco Way',70000);
INSERT INTO "Position" VALUES(5,4,'Position E','Web Developer','2/7 London Street',75000);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Company',5);
INSERT INTO "sqlite_sequence" VALUES('Department',5);
INSERT INTO "sqlite_sequence" VALUES('Position',5);
INSERT INTO "sqlite_sequence" VALUES('Employee',5);
COMMIT;
