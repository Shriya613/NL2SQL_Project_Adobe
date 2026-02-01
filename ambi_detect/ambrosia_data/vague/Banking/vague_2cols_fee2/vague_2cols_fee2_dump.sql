BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountNumber INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    AccountType TEXT,
    OpenDate DATE,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,1,'Checking','2021-06-15');
INSERT INTO "Accounts" VALUES(2,2,'Savings','2021-07-01');
CREATE TABLE CheckingAccounts (
    AccountNumber INTEGER PRIMARY KEY,
    Monthly_Maintenance_Fee DECIMAL(7,2),
    Overdraft_Charges DECIMAL(7,2),
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "CheckingAccounts" VALUES(1,5,30);
INSERT INTO "CheckingAccounts" VALUES(2,4.5,25);
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    DateOfBirth DATE
, fullname TEXT);
INSERT INTO "Customers" VALUES(1,'1986-03-10','John Doe');
INSERT INTO "Customers" VALUES(2,'1991-05-25','Jane Smith');
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    PrincipalAmount DECIMAL(10,2),
    AnnualInterestRate DECIMAL(5,2),
    RepaymentTermYears INTEGER,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Loans" VALUES(1,1,20000,0.05,5);
INSERT INTO "Loans" VALUES(2,2,15000,0.04,3);
CREATE TABLE SavingsAccounts (
    AccountNumber INTEGER PRIMARY KEY,
    InterestRate DECIMAL(5,2),
    MinimumBalance DECIMAL(10,2),
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "SavingsAccounts" VALUES(1,0.1,500);
INSERT INTO "SavingsAccounts" VALUES(2,0.05,1000);
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    Amount DECIMAL(10,2),
    TransactionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Transactions" VALUES(1,1,1500,'2024-02-19 14:39:46');
INSERT INTO "Transactions" VALUES(2,1,-500,'2024-02-19 14:39:46');
COMMIT;
