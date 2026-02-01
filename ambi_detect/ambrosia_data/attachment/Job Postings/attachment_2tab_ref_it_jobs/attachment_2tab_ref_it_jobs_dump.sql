BEGIN TRANSACTION;
CREATE TABLE Employee(
    ID INTEGER PRIMARY KEY,
    First_Name TEXT,
    Last_Name TEXT,
    Position TEXT,
    Salary DECIMAL(10,2),
    Joining_Date DATE);
INSERT INTO "Employee" VALUES(1,'Alice','Smith','Software Engineer',75000,'2019-06-01');
INSERT INTO "Employee" VALUES(2,'Bob','Williams','Data Analyst',65000,'2019-05-01');
INSERT INTO "Employee" VALUES(3,'Carol','Johnson','Product Manager',85000,'2019-07-01');
INSERT INTO "Employee" VALUES(4,'David','Brown','UI/UX Designer',60000,'2019-08-01');
INSERT INTO "Employee" VALUES(5,'Eve','Jones','QA Tester',55000,'2019-09-01');
CREATE TABLE IT_Support_Technician(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Contact_Email TEXT,
    Phone_Number TEXT,
    Hire_Date DATE,
    Operating_System_ID INTEGER,
    FOREIGN KEY(Operating_System_ID) REFERENCES Operating_System_Knowledge(ID));
INSERT INTO "IT_Support_Technician" VALUES(1,'John Doe','john@example.com','+1 (123) 456-7890','2020-01-01',1);
INSERT INTO "IT_Support_Technician" VALUES(2,'Jane Smith','jane@example.com','+1 (123) 456-7890','2020-01-01',2);
CREATE TABLE Network_Administrator(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Contact_Email TEXT,
    Phone_Number TEXT,
    Hire_Date DATE,
    Operating_System_ID INTEGER,
    FOREIGN KEY(Operating_System_ID) REFERENCES Operating_System_Knowledge(ID));
INSERT INTO "Network_Administrator" VALUES(1,'Mike Johnson','mike@example.com','+1 (123) 456-7890','2020-01-01',1);
INSERT INTO "Network_Administrator" VALUES(2,'Emily Davis','emily@example.com','+1 (123) 456-7890','2020-01-01',2);
CREATE TABLE Operating_System_Knowledge(
    ID INTEGER PRIMARY KEY,
    OS_Name TEXT);
INSERT INTO "Operating_System_Knowledge" VALUES(1,'Windows');
INSERT INTO "Operating_System_Knowledge" VALUES(2,'Linux');
COMMIT;
