BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountNumber INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    AccountType TEXT,
    OpenDate DATE,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1001,1,'Checking','2021-06-01');
INSERT INTO "Accounts" VALUES(1002,1,'Savings','2021-06-01');
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','1980-06-15');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1973-08-23');
CREATE TABLE Fees (
    FeeID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    Fees DECIMAL(5,2),
    EffectiveFrom DATE,
    EffectiveTo DATE,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Fees" VALUES(1001,1001,2,'2021-06-01','2022-06-01');
INSERT INTO "Fees" VALUES(1002,1001,2.5,'2022-06-01','2023-06-01');
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    PrincipalAmount DECIMAL(10,2),
    RateOfInterest DECIMAL(4,2),
    TenureMonths INTEGER,
    StartDate DATE,
    EndDate DATE,
    RepaymentMethod TEXT,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Loans" VALUES(1001,1001,1000,8,12,'2021-06-01','2023-06-01','Equal monthly installments');
INSERT INTO "Loans" VALUES(1002,1001,800,7,18,'2021-06-01','2023-06-01','Weekly repayments');
CREATE TABLE Overdrafts (
    OverdraftID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    LimitAmount DECIMAL(10,2),
    InterestRate DECIMAL(4,2),
    ApprovalDate DATE,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Overdrafts" VALUES(1001,1001,500,10,'2021-06-01');
INSERT INTO "Overdrafts" VALUES(1002,1001,800,12,'2021-06-01');
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    Amount DECIMAL(10,2),
    TransactionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Transactions" VALUES(1,1001,-200,'2021-06-01 08:00:00');
INSERT INTO "Transactions" VALUES(2,1001,150,'2021-06-01 09:00:00');
COMMIT;
