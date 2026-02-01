BEGIN TRANSACTION;
CREATE TABLE Data_Analyst(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Tools_Used TEXT, -- comma separated if needed
    Analytical_Skills INT ,
    Problem_Solving_Abilities INT ,
    Salary TEXT);
INSERT INTO "Data_Analyst" VALUES(1,'Excel, Power BI',10,10,'$70,000');
INSERT INTO "Data_Analyst" VALUES(2,'Tableau, SAS',8,8,'$60,000');
INSERT INTO "Data_Analyst" VALUES(3,'R, Python',12,12,'$75,000');
INSERT INTO "Data_Analyst" VALUES(4,'SQL Server, SSIS',5,5,'$42,000');
INSERT INTO "Data_Analyst" VALUES(5,'Jupyter Notebook, KNIME',10,10,'$80,000');
CREATE TABLE HR_Specialist(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Industry_Experience INT ,
    Certifications TEXT, -- comma separated if needed
    Recruitment_Methods TEXT,
    Salary TEXT);
INSERT INTO "HR_Specialist" VALUES(1,'5 years','SHRM CP, PHR','Direct Hire, Temp Staffing','$60,000');
INSERT INTO "HR_Specialist" VALUES(2,'10 years','SPHR, GPHR','Headhunting, Job Boards','$70,000');
INSERT INTO "HR_Specialist" VALUES(3,'3 years','CPHR, CHRP','Onboarding, Offboarding','$40,000');
INSERT INTO "HR_Specialist" VALUES(4,'7 years','CIH, CEBS','Employee Relations','$55,000');
INSERT INTO "HR_Specialist" VALUES(5,'8 years','GPHR, SHRM SCP','Payroll Management','$65,000');
CREATE TABLE IT_Consultant(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "Type" TEXT,
    Projects_Worked INT ,
    Languages_Known TEXT, -- comma separated if needed
    Salary TEXT);
INSERT INTO "IT_Consultant" VALUES(1,'Application Developer',10,'Java, Python','$70,000');
INSERT INTO "IT_Consultant" VALUES(2,'Business Analyst',6,'SQL, R','$60,000');
INSERT INTO "IT_Consultant" VALUES(3,'Cloud Solution Consultant',8,'AWS, Azure','$75,000');
INSERT INTO "IT_Consultant" VALUES(4,'Technology Strategist',5,'Power BI, Tableau','$55,000');
INSERT INTO "IT_Consultant" VALUES(5,'Agile Coach',12,'Scrum, Kanban','$80,000');
CREATE TABLE Network_Admin(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Position TEXT,
    Years_of_Experience INT ,
    Degree_Level TEXT ,
    Salary TEXT);
INSERT INTO "Network_Admin" VALUES(1,'Network Administrator',5,'Bachelor','$70,000');
INSERT INTO "Network_Admin" VALUES(2,'System Administrator',8,'Master','$65,000');
INSERT INTO "Network_Admin" VALUES(3,'IT Support',3,'Associate','$42,000');
INSERT INTO "Network_Admin" VALUES(4,'Data Center Technician',10,'Professional','$72,000');
INSERT INTO "Network_Admin" VALUES(5,'Cybersecurity Expert',12,'PhD','$90,000');
CREATE TABLE Software_Engineer(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Programming_Languages TEXT, -- comma separated if needed
    Experience_in_Teamwork INT ,
    Specialization TEXT,
    Salary TEXT);
INSERT INTO "Software_Engineer" VALUES(1,'Python, JavaScript, C++',10,'Full Stack','$70,000');
INSERT INTO "Software_Engineer" VALUES(2,'Ruby on Rails, Java',5,'Backend','$50,000');
INSERT INTO "Software_Engineer" VALUES(3,'Android, Swift',8,'Mobile','$75,000');
INSERT INTO "Software_Engineer" VALUES(4,'Go, PHP',6,'DevOps','$60,000');
INSERT INTO "Software_Engineer" VALUES(5,'Node.js, Angular',12,'Frontend','$85,000');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Network_Admin',5);
INSERT INTO "sqlite_sequence" VALUES('IT_Consultant',5);
INSERT INTO "sqlite_sequence" VALUES('Software_Engineer',5);
INSERT INTO "sqlite_sequence" VALUES('HR_Specialist',5);
INSERT INTO "sqlite_sequence" VALUES('Data_Analyst',5);
COMMIT;
