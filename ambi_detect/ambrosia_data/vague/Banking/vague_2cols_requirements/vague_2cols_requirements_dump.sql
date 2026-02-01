BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountNumber INTEGER PRIMARY KEY AUTOINCREMENT,
    Balance REAL DEFAULT 0.0,
    InterestRate REAL DEFAULT 0.0,
    OpenDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Status TEXT ,
    CustomerID INTEGER,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Accounts" VALUES(1,1000.0,'2.5%','2021-06-01','Active',1);
INSERT INTO "Accounts" VALUES(2,2000.0,'1.5%','2021-07-01','Inactive',2);
CREATE TABLE Branches (
    BranchID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT,
    ManagerID INTEGER,
    FOREIGN KEY(ManagerID) REFERENCES Employees(EmployeeID));
INSERT INTO "Branches" VALUES(1,'Main Street Branch','1 Main St, Anytown','555-123-4567',NULL);
INSERT INTO "Branches" VALUES(2,'Southside Branch','123 South Rd, Anycity','555-987-6543',1);
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    EmailAddress TEXT UNIQUE);
INSERT INTO "Customers" VALUES(1,'John Doe','Doe','2000-01-01','johndoe@email.com');
INSERT INTO "Customers" VALUES(2,'Jane Smith','Smith','1998-03-27','janesmith@email.com');
CREATE TABLE EligibilityAndDocuments (
    ApplicationID INTEGER PRIMARY KEY AUTOINCREMENT,
    Eligibility_Criteria TEXT,
    Documentation_Requirements TEXT,
    Business_Loan_Application BOOLEAN,
    LoanID INTEGER,
    FOREIGN KEY(LoanID) REFERENCES Loans(LoanID));
INSERT INTO "EligibilityAndDocuments" VALUES(1,'Excellent credit history.','Proof of income, tax returns.',0,1);
INSERT INTO "EligibilityAndDocuments" VALUES(2,'Good credit history.','Bank statements, pay stubs.',1,2);
CREATE TABLE Employees (
    EmployeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    HireDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Role TEXT,
    Salary REAL,
    BranchID INTEGER,
    FOREIGN KEY(BranchID) REFERENCES Branches(BranchID));
INSERT INTO "Employees" VALUES(1,'Alice Johnson','Johnson','2021-01-01','Teller',40000.0,1);
INSERT INTO "Employees" VALUES(2,'Michael Brown','Brown','2021-01-15','Branch manager',50000.0,1);
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    Amount REAL,
    Term INT,
    Rate REAL,
    ApprovalDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Status TEXT ,
    CustomerID INTEGER,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Loans" VALUES(1,50000.0,36,'5.5%','2021-06-10','Approved',1);
INSERT INTO "Loans" VALUES(2,40000.0,24,'4.0%','2021-07-05','Pending',2);
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    Type TEXT ,
    Amount REAL,
    Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    AccountNumber INTEGER,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber));
INSERT INTO "Transactions" VALUES(1,'Salary deposit','Credit',3000.0,'2021-06-15',1);
INSERT INTO "Transactions" VALUES(2,'Utility payment','Debit',-200.0,'2021-06-20',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',2);
INSERT INTO "sqlite_sequence" VALUES('Accounts',2);
INSERT INTO "sqlite_sequence" VALUES('Loans',2);
INSERT INTO "sqlite_sequence" VALUES('EligibilityAndDocuments',2);
INSERT INTO "sqlite_sequence" VALUES('Transactions',2);
INSERT INTO "sqlite_sequence" VALUES('Employees',2);
INSERT INTO "sqlite_sequence" VALUES('Branches',2);
COMMIT;
