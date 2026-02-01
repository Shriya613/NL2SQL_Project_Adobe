BEGIN TRANSACTION;
CREATE TABLE "Checking_Accounts"(
    AccountID INTEGER PRIMARY KEY, `AccountNumber` INTEGER,
    CustomerName TEXT,
    OpenDate DATE,
    Balance REAL DEFAULT 0.0,
    OverdraftLimit REAL DEFAULT -500.0,
    Interest_Rate TEXT);
INSERT INTO "Checking_Accounts" VALUES(1,123,'John Doe','2023-01-01',500.0,250.0,'1%');
INSERT INTO "Checking_Accounts" VALUES(2,345,'Jane Smith','2023-01-01',750.0,350.0,'2%');
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email TEXT UNIQUE);
INSERT INTO "Customers" VALUES(1,'John','Doe','1990-01-01','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1992-01-01','jane.smith@example.com');
CREATE TABLE Loans(
    LoanID INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    Amount BIGINT,
    StartDate DATE,
    EndDate DATE,
    Rate DECIMAL(8,2),
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Loans" VALUES(1,1,5000,'2023-01-01','2026-01-01',4.5);
INSERT INTO "Loans" VALUES(2,2,10000,'2023-01-01','2028-01-01',4.75);
CREATE TABLE "Savings_Accounts"(
    AccountID INTEGER PRIMARY KEY, "AccountNumber" INTEGER,
    CustomerName TEXT,
    OpenDate DATE,
    Balance REAL DEFAULT 0.0,
    Interest_Rate TEXT);
INSERT INTO "Savings_Accounts" VALUES(1,567,'John Doe','2023-01-01',1000.0,'1%');
INSERT INTO "Savings_Accounts" VALUES(2,789,'Jane Smith','2023-01-01',2000.0,'2%');
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY,
    AccountID INTEGER,
    Type TEXT ,
    Amount REAL,
    DateAndTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(AccountID) REFERENCES Savings_Accounts(AccountID),
    FOREIGN KEY(AccountID) REFERENCES Checking_Accounts(AccountID));
INSERT INTO "Transactions" VALUES(1,1,'Deposit',500.0,'2023-01-01 12:00:00');
INSERT INTO "Transactions" VALUES(2,2,'Withdrawal',250.0,'2023-01-01 13:00:00');
COMMIT;
