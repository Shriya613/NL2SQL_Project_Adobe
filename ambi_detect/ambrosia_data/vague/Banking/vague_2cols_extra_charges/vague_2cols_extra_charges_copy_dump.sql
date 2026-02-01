BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER REFERENCES Customers(CustomerID),
    AccountType TEXT ,
    Balance REAL ,
    OpenDate DATE
);
INSERT INTO "Accounts" VALUES(1,1,'Checking',1000.0,'2021-01-01');
INSERT INTO "Accounts" VALUES(2,2,'Savings',500.0,'2021-01-02');
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT,
    PhoneNumber VARCHAR(15)
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com','+1-503-789-1234');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com','+1-503-789-5678');
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER REFERENCES Customers(CustomerID),
    LoanAmount REAL ,
    InterestRate DECIMAL(5,2) ,
    TermYears INTEGER ,
    ApprovalDate DATE
);
INSERT INTO "Loans" VALUES(1,1,10000.0,3.5,5,'2021-01-01');
INSERT INTO "Loans" VALUES(2,2,5000.0,4,3,'2021-01-02');
CREATE TABLE Overdrafts (
    OverdraftID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER REFERENCES Accounts(AccountID),
    LimitAmount REAL ,
    InterestRate DECIMAL(5,2) ,
    Fees PERCENTAGE 
);
INSERT INTO "Overdrafts" VALUES(1,1,500.0,4.5,0.1);
INSERT INTO "Overdrafts" VALUES(2,2,1000.0,4,0.15);
CREATE TABLE Rewards (
    RewardID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER REFERENCES Accounts(AccountID),
    Points ,
    RedemptionValue REAL 
);
INSERT INTO "Rewards" VALUES(1,1,100,10.0);
INSERT INTO "Rewards" VALUES(2,2,50,5.0);
CREATE TABLE "Transactions" (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    FromAccountID INTEGER REFERENCES Accounts(AccountID),
    ToAccountID INTEGER REFERENCES Accounts(AccountID),
    Amount REAL ,
    Date DATE,
    Extra_Charges REAL,
    Transactions INT DEFAULT 1
);
INSERT INTO "Transactions" VALUES(1,1,2,100.0,'2021-01-01',2.0,1);
INSERT INTO "Transactions" VALUES(2,2,3,200.0,'2021-01-02',4.0,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Accounts',2);
INSERT INTO "sqlite_sequence" VALUES('Loans',2);
INSERT INTO "sqlite_sequence" VALUES('Overdrafts',2);
INSERT INTO "sqlite_sequence" VALUES('Rewards',2);
INSERT INTO "sqlite_sequence" VALUES('Transactions',2);
COMMIT;
