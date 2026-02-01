BEGIN TRANSACTION;
CREATE TABLE "Dentists"(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    DentalSpecialty TEXT,
    JoiningDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    EducationID INTEGER, FullName TEXT,
    FOREIGN KEY(EducationID) REFERENCES Education(ID)
);
INSERT INTO "Dentists" VALUES(1,'Orthodontist','2024-03-13 12:00:50',1,'Alice Smith');
INSERT INTO "Dentists" VALUES(2,'Orthodontist','2024-03-13 12:00:50',2,'Bob Johnson');
INSERT INTO "Dentists" VALUES(3,'Periodontist','2024-03-13 12:00:50',1,'Charlie Williams');
INSERT INTO "Dentists" VALUES(4,'Periodontist','2024-03-13 12:00:50',2,'David Brown');
CREATE TABLE Education(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Degree TEXT
);
INSERT INTO "Education" VALUES(1,'Doctor of Dental Surgery');
INSERT INTO "Education" VALUES(2,'Master of Science in Dentistry');
CREATE TABLE Employees(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Email TEXT,
    PhoneNumber INT,
    HireDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    JobID INTEGER,
    FOREIGN KEY(JobID) REFERENCES Jobs(ID) ON DELETE SET NULL
);
INSERT INTO "Employees" VALUES(1,'John Doe','john@gmail.com',1234567890,'2024-03-13 12:00:50',1);
INSERT INTO "Employees" VALUES(2,'Jane Smith','jane@yahoo.com',9876543210,'2024-03-13 12:00:50',2);
INSERT INTO "Employees" VALUES(3,'Michael Johnson','mike@outlook.com',5555555555,'2024-03-13 12:00:50',3);
INSERT INTO "Employees" VALUES(4,'Emma Williams','emma@hotmail.com',7777777777,'2024-03-13 12:00:50',4);
CREATE TABLE Jobs(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    Location TEXT,
    CompanyName TEXT,
    PostedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FullTimePartTime TEXT 
);
INSERT INTO "Jobs" VALUES(1,'Orthodontist','An experienced Orthodontist required.','New York','ABCD Inc','2024-03-13 12:00:50','Full Time');
INSERT INTO "Jobs" VALUES(2,'General Dentist','A General Dentist needed for our clinic.','California','XYZ Clinic','2024-03-13 12:00:50','Full Time');
INSERT INTO "Jobs" VALUES(3,'Periodontist','A qualified Periodontist wanted for our hospital.','Texas','LMN Hospital','2024-03-13 12:00:50','Full Time');
INSERT INTO "Jobs" VALUES(4,'Dental Assistant','Experienced Dental Assistants needed.','Florida','OPQ Clinic','2024-03-13 12:00:50','Part Time');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Education',2);
INSERT INTO "sqlite_sequence" VALUES('Jobs',4);
INSERT INTO "sqlite_sequence" VALUES('Employees',4);
INSERT INTO "sqlite_sequence" VALUES('Dentists',4);
COMMIT;
