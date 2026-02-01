BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountNumber INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Balance REAL DEFAULT 0,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,1,500.0);
INSERT INTO "Accounts" VALUES(2,2,1000.0);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','1980-06-17','john.doe@gmail.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1985-03-19','jane.smith@yahoo.com');
CREATE TABLE Insurance(
    InsuranceID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    InsuranceType TEXT,
    PaymentFrequency TEXT ,
    PremiumAmount REAL,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Insurance" VALUES(1,1,'Pet Insurance','Monthly',15.0,'2021-01-01','2021-12-31');
INSERT INTO "Insurance" VALUES(2,1,'Pet Insurance','Quarterly',45.0,'2021-01-01','2021-12-31');
INSERT INTO "Insurance" VALUES(3,2,'Disability Insurance','Monthly',20.0,'2021-01-01','2021-12-31');
INSERT INTO "Insurance" VALUES(4,2,'Disability Insurance','Annually',200.0,'2021-01-01','2022-12-31');
CREATE TABLE Loans(
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Amount BIGINT,
    InterestRate DECIMAL(5,2) ,
    TermYears INTEGER ,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Loans" VALUES(1,1,20000,0.05,5);
INSERT INTO "Loans" VALUES(2,2,15000,0.04,3);
CREATE TABLE Payments(
    PaymentID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountNumber INTEGER,
    PaymentDate DATE,
    PaymentAmount REAL,
    CheckNumber INTEGER,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Payments" VALUES(1,1,'2021-01-15',1000.0,12345);
INSERT INTO "Payments" VALUES(2,1,'2021-02-15',1500.0,67890);
INSERT INTO "Payments" VALUES(3,2,'2021-01-15',500.0,123456);
INSERT INTO "Payments" VALUES(4,2,'2021-02-15',500.0,67890);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',2);
INSERT INTO "sqlite_sequence" VALUES('Accounts',2);
INSERT INTO "sqlite_sequence" VALUES('Loans',2);
INSERT INTO "sqlite_sequence" VALUES('Insurance',4);
INSERT INTO "sqlite_sequence" VALUES('Payments',4);
COMMIT;
