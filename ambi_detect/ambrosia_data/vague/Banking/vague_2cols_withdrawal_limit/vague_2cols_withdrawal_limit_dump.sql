BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountNumber INTEGER PRIMARY KEY,
    AccountType TEXT ,
    Balance REAL DEFAULT 0,
    OpenDate DATE,
    CustomerID INTEGER,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,'Checking',1000.0,'2023-01-01',1);
INSERT INTO "Accounts" VALUES(2,'Savings',5000.0,'2023-02-01',2);
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com');
CREATE TABLE "Limits" (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Daily_Limit REAL,
    Transaction_Limit REAL,
    Atms INTEGER,
    AccountNumber INTEGER,
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Limits" VALUES(1,3000.0,600.0,2,1);
INSERT INTO "Limits" VALUES(2,4000.0,800.0,3,2);
CREATE TABLE Overdrafts (
    OverdraftID INTEGER PRIMARY KEY AUTOINCREMENT,
    ApprovalStatus BOOLEAN,
    LimitAmount REAL,
    StartDate DATE,
    EndDate DATE,
    AccountNumber INTEGER,
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Overdrafts" VALUES(1,1,500.0,'2023-04-01','2023-05-01',1);
INSERT INTO "Overdrafts" VALUES(2,0,750.0,'2023-05-01','2023-06-01',2);
CREATE TABLE Rewards (
    RewardID INTEGER PRIMARY KEY AUTOINCREMENT,
    Points ,
    ExpiryDate DATE,
    AccountNumber INTEGER,
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Rewards" VALUES(1,100,'2024-01-01',1);
INSERT INTO "Rewards" VALUES(2,200,'2024-02-01',2);
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Amount REAL,
    DateAndTime TIMESTAMP,
    Description TEXT,
    AccountNumber INTEGER,
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Transactions" VALUES(1,200.0,'2023-03-01 10:30:00','Groceries',1);
INSERT INTO "Transactions" VALUES(2,-150.0,'2023-03-02 14:15:00','ATM Withdrawal',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Limits',2);
INSERT INTO "sqlite_sequence" VALUES('Transactions',2);
INSERT INTO "sqlite_sequence" VALUES('Overdrafts',2);
INSERT INTO "sqlite_sequence" VALUES('Rewards',2);
COMMIT;
