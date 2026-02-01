BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Balance REAL,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Accounts" VALUES(1,1,1000.0);
INSERT INTO "Accounts" VALUES(2,2,2000.0);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Address TEXT);
INSERT INTO "Customers" VALUES(1,'John','Doe','1970-01-01','123 Main St');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1980-06-15','456 Elm St');
CREATE TABLE Deposits(
    DepositID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    Amount REAL,
    TransactionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(AccountID) REFERENCES Accounts(AccountID));
INSERT INTO "Deposits" VALUES(1,1,500.0,'2024-03-08 00:00:42');
INSERT INTO "Deposits" VALUES(2,2,1000.0,'2024-03-08 00:00:42');
CREATE TABLE Loans(
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    StockMarketOfferings TEXT,
    RaisedCapitalAmount VARCHAR(128),
    InterestRate REAL,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Loans" VALUES(1,1,'IPO','50000000',5.0,NULL,NULL);
INSERT INTO "Loans" VALUES(2,2,'IPO','60000000',4.5,NULL,NULL);
INSERT INTO "Loans" VALUES(3,1,'Secondary Offering','50000000',5.5,NULL,NULL);
INSERT INTO "Loans" VALUES(4,2,'Secondary Offering','60000000',4.0,NULL,NULL);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',2);
INSERT INTO "sqlite_sequence" VALUES('Accounts',2);
INSERT INTO "sqlite_sequence" VALUES('Deposits',2);
INSERT INTO "sqlite_sequence" VALUES('Loans',4);
COMMIT;
