BEGIN TRANSACTION;
CREATE TABLE Applicant_Details(
    ApplicantID INTEGER PRIMARY KEY AUTOINCREMENT,
    Fullname TEXT,
    Age INTEGER,
    Occupation TEXT);
INSERT INTO "Applicant_Details" VALUES(1,'Jack Black',30,'Engineer');
INSERT INTO "Applicant_Details" VALUES(2,'Jill Green',28,'Doctor');
INSERT INTO "Applicant_Details" VALUES(3,'Steve Blue',35,'Teacher');
INSERT INTO "Applicant_Details" VALUES(4,'Lisa Red',40,'Accountant');
INSERT INTO "Applicant_Details" VALUES(5,'Chris Yellow',29,'Lawyer');
CREATE TABLE Employee_Details(
    EmployeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    Fullname TEXT,
    Position TEXT);
INSERT INTO "Employee_Details" VALUES(1,'John Doe','Manager');
INSERT INTO "Employee_Details" VALUES(2,'Jane Smith','Officer');
INSERT INTO "Employee_Details" VALUES(3,'Tom Brown','Assistant Manager');
INSERT INTO "Employee_Details" VALUES(4,'Alice Johnson','Clerk');
INSERT INTO "Employee_Details" VALUES(5,'Bob White','Analyst');
CREATE TABLE Loan_Applications(
    ApplicationID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerName TEXT,
    AmountMoney REAL);
INSERT INTO "Loan_Applications" VALUES(1,'Jack Black',10000.0);
INSERT INTO "Loan_Applications" VALUES(2,'Jill Green',15000.0);
INSERT INTO "Loan_Applications" VALUES(3,'Steve Blue',20000.0);
INSERT INTO "Loan_Applications" VALUES(4,'Lisa Red',25000.0);
INSERT INTO "Loan_Applications" VALUES(5,'Chris Yellow',30000.0);
CREATE TABLE Loan_Applications_Requirements(
    AppID INTEGER,
    ReqID INTEGER,
    FOREIGN KEY(AppID) REFERENCES Loan_Applications(ApplicationID),
    FOREIGN KEY(ReqID) REFERENCES Requirements(RequirementID));
INSERT INTO "Loan_Applications_Requirements" VALUES(1,1);
INSERT INTO "Loan_Applications_Requirements" VALUES(2,1);
INSERT INTO "Loan_Applications_Requirements" VALUES(3,1);
INSERT INTO "Loan_Applications_Requirements" VALUES(4,1);
INSERT INTO "Loan_Applications_Requirements" VALUES(5,1);
INSERT INTO "Loan_Applications_Requirements" VALUES(5,3);
CREATE TABLE Requirements(
    RequirementID INTEGER PRIMARY KEY AUTOINCREMENT,
    RequirementDescription TEXT);
INSERT INTO "Requirements" VALUES(1,'Employment Verification');
INSERT INTO "Requirements" VALUES(2,'Income Statement');
INSERT INTO "Requirements" VALUES(3,'Credit History Report');
INSERT INTO "Requirements" VALUES(4,'Tax Returns');
INSERT INTO "Requirements" VALUES(5,'Property Valuation');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Requirements',5);
INSERT INTO "sqlite_sequence" VALUES('Employee_Details',5);
INSERT INTO "sqlite_sequence" VALUES('Applicant_Details',5);
INSERT INTO "sqlite_sequence" VALUES('Loan_Applications',5);
COMMIT;
