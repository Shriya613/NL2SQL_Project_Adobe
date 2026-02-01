BEGIN TRANSACTION;
CREATE TABLE Applicant (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Full_name TEXT,
    Email TEXT,
    Phone_number TEXT,
    Resume_url TEXT,
    Date_applied TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Job_posting_id INTEGER,
    FOREIGN KEY(Job_posting_id) REFERENCES Job_Posting(ID)
);
INSERT INTO "Applicant" VALUES(1,'John Doe','johndoe@email.com','+1 234-567-8901','https://www.example.com/resume-john-doe','2024-02-19 12:27:01',1);
INSERT INTO "Applicant" VALUES(2,'Jane Smith','janesmith@email.com','+1 345-678-9012','https://www.example.com/resume-jane-smith','2024-02-19 12:27:01',2);
INSERT INTO "Applicant" VALUES(3,'Alice Johnson','alicejohnson@email.com','+1 456-789-0123','https://www.example.com/resume-alice-johnson','2024-02-19 12:27:01',3);
INSERT INTO "Applicant" VALUES(4,'Bob Williams','bobwilliams@email.com','+1 567-890-1234','https://www.example.com/resume-bob-williams','2024-02-19 12:27:01',4);
INSERT INTO "Applicant" VALUES(5,'Emma Brown','emmabrown@email.com','+1 678-901-2345','https://www.example.com/resume-emma-brown','2024-02-19 12:27:01',5);
CREATE TABLE Department (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Department" VALUES(1,'Engineering');
INSERT INTO "Department" VALUES(2,'Marketing');
INSERT INTO "Department" VALUES(3,'Human Resources');
INSERT INTO "Department" VALUES(4,'Finance');
INSERT INTO "Department" VALUES(5,'IT');
CREATE TABLE Employee (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    First_Name TEXT,
    Last_Name TEXT,
    Email TEXT,
    Phone_Number TEXT,
    Hire_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "Employee" VALUES(1,'John','Doe','john.doe@example.com','+1 234-567-8901','2024-02-19 12:27:01');
INSERT INTO "Employee" VALUES(2,'Jane','Smith','jane.smith@example.com','+1 345-678-9012','2024-02-19 12:27:01');
INSERT INTO "Employee" VALUES(3,'Alice','Johnson','alice.johnson@example.com','+1 456-789-0123','2024-02-19 12:27:01');
INSERT INTO "Employee" VALUES(4,'Bob','Williams','bob.williams@example.com','+1 567-890-1234','2024-02-19 12:27:01');
INSERT INTO "Employee" VALUES(5,'Emma','Brown','emma.brown@example.com','+1 678-901-2345','2024-02-19 12:27:01');
CREATE TABLE Internship_Position (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Position_Title TEXT,
    Description TEXT,
    Duration INT
);
INSERT INTO "Internship_Position" VALUES(1,'Software Engineering Intern','Help develop software applications.',12);
INSERT INTO "Internship_Position" VALUES(2,'Graphic Design Intern','Create visual designs for marketing materials.',12);
INSERT INTO "Internship_Position" VALUES(3,'HR Intern','Assist with recruitment and onboarding processes.',12);
INSERT INTO "Internship_Position" VALUES(4,'Financial Analyst Intern','Analyze financial data and support decision-making.',12);
INSERT INTO "Internship_Position" VALUES(5,'Network Engineer Intern','Support network infrastructure and troubleshoot issues.',12);
CREATE TABLE Job_Posting (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    Required_Skills TEXT,
    Salary REAL ,
    Application_Deadline TIMESTAMP,
    Office_ID INTEGER,
    Departments_ID INTEGER,
    Internships_ID INTEGER,
    FOREIGN KEY(Office_ID) REFERENCES Office(ID),
    FOREIGN KEY(Departments_ID) REFERENCES Department(ID),
    FOREIGN KEY(Internships_ID) REFERENCES Internship_Position(ID)
);
INSERT INTO "Job_Posting" VALUES(1,'Software Developer','Design, code, and maintain software systems.','JavaScript, HTML, CSS',80000.0,'2023-05-31 23:59:59',1,1,NULL);
INSERT INTO "Job_Posting" VALUES(2,'Digital Marketing Specialist','Plan and execute digital marketing campaigns.','SEO, SEM, Social Media',60000.0,'2023-05-31 23:59:59',2,2,NULL);
INSERT INTO "Job_Posting" VALUES(3,'Recruiter','Source, screen, and interview candidates.','Applicant Tracking Systems',50000.0,'2023-05-31 23:59:59',3,3,NULL);
INSERT INTO "Job_Posting" VALUES(4,'Accountant','Prepare financial reports and analyze data.','Excel, QuickBooks',65000.0,'2023-05-31 23:59:59',4,4,NULL);
INSERT INTO "Job_Posting" VALUES(5,'System Administrator','Install, configure, and maintain computer systems.','Linux, Windows Server',70000.0,'2023-05-31 23:59:59',5,5,NULL);
CREATE TABLE "Office" (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Location TEXT);
INSERT INTO "Office" VALUES(1,'123 Main St');
INSERT INTO "Office" VALUES(2,'456 Elm St');
INSERT INTO "Office" VALUES(3,'789 Oak St');
INSERT INTO "Office" VALUES(4,'012 Pine St');
INSERT INTO "Office" VALUES(5,'321 Fir St');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Department',5);
INSERT INTO "sqlite_sequence" VALUES('Internship_Position',5);
INSERT INTO "sqlite_sequence" VALUES('Employee',5);
INSERT INTO "sqlite_sequence" VALUES('Job_Posting',5);
INSERT INTO "sqlite_sequence" VALUES('Applicant',5);
INSERT INTO "sqlite_sequence" VALUES('Office',5);
COMMIT;
