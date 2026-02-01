BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Balance REAL DEFAULT 0.0,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,1,1000.0);
INSERT INTO "Accounts" VALUES(2,2,500.0);
CREATE TABLE "Banks" (
    BankID INTEGER PRIMARY KEY,
    Number_of_Branches INTEGER,
    Countries_served TEXT,
    Bank TEXT
);
INSERT INTO "Banks" VALUES(1,100,'USA and Canada','ABC Bank');
INSERT INTO "Banks" VALUES(2,200,'Europe','XYZ Bank');
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com');
CREATE TABLE "Employees" (
    EmployeeID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Position TEXT,
    Salary REAL,
    "BankID" INTEGER,
    FOREIGN KEY("BankID") REFERENCES Banks(BankID)
);
INSERT INTO "Employees" VALUES(1,'Alice','Wright','Manager',50000.0,1);
INSERT INTO "Employees" VALUES(2,'Bob','Hanson','Teller',35000.0,2);
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Amount BIGINT,
    InterestRate DECIMAL(5,2),
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Loans" VALUES(1,1,50000,3.5,'2023-01-01','2026-12-31');
INSERT INTO "Loans" VALUES(2,2,75000,3,'2023-02-01','2028-01-31');
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Type TEXT ,
    Amount REAL,
    FOREIGN KEY(AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "Transactions" VALUES(1,1,'2024-02-19 15:10:48','Deposit',100.0);
INSERT INTO "Transactions" VALUES(2,2,'2024-02-19 15:10:48','Withdrawal',-50.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Accounts',2);
INSERT INTO "sqlite_sequence" VALUES('Loans',2);
INSERT INTO "sqlite_sequence" VALUES('Transactions',2);
COMMIT;
