BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountNumber INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Balance REAL,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Accounts" VALUES(1,1,1000.0);
INSERT INTO "Accounts" VALUES(2,2,800.0);
INSERT INTO "Accounts" VALUES(3,3,900.0);
INSERT INTO "Accounts" VALUES(4,4,700.0);
INSERT INTO "Accounts" VALUES(5,5,600.0);
CREATE TABLE Bank_Statements(
    StatementID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountNumber INTEGER,
    StartDate DATETIME,
    EndDate DATETIME,
    TotalAmount REAL,
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber));
INSERT INTO "Bank_Statements" VALUES(1,1,'2021-01-01','2021-01-31',250.0);
INSERT INTO "Bank_Statements" VALUES(2,2,'2021-02-01','2021-02-28',260.0);
INSERT INTO "Bank_Statements" VALUES(3,3,'2021-03-01','2021-03-31',270.0);
INSERT INTO "Bank_Statements" VALUES(4,4,'2021-04-01','2021-04-30',280.0);
INSERT INTO "Bank_Statements" VALUES(5,5,'2021-05-01','2021-05-31',290.0);
CREATE TABLE Bank_Statements_Transactions(
    StatementTransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    StatementID INTEGER,
    TransactionID INTEGER,
    FOREIGN KEY (StatementID) REFERENCES Bank_Statements(StatementID),
    FOREIGN KEY (TransactionID) REFERENCES Transactions(TransactionID));
INSERT INTO "Bank_Statements_Transactions" VALUES(1,1,1);
INSERT INTO "Bank_Statements_Transactions" VALUES(2,1,2);
INSERT INTO "Bank_Statements_Transactions" VALUES(3,2,3);
INSERT INTO "Bank_Statements_Transactions" VALUES(4,2,4);
INSERT INTO "Bank_Statements_Transactions" VALUES(5,3,5);
INSERT INTO "Bank_Statements_Transactions" VALUES(6,3,6);
INSERT INTO "Bank_Statements_Transactions" VALUES(7,4,7);
INSERT INTO "Bank_Statements_Transactions" VALUES(8,4,8);
INSERT INTO "Bank_Statements_Transactions" VALUES(9,5,9);
INSERT INTO "Bank_Statements_Transactions" VALUES(10,5,10);
INSERT INTO "Bank_Statements_Transactions" VALUES(11,2,1);
INSERT INTO "Bank_Statements_Transactions" VALUES(12,3,1);
INSERT INTO "Bank_Statements_Transactions" VALUES(13,4,1);
INSERT INTO "Bank_Statements_Transactions" VALUES(14,5,1);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT);
INSERT INTO "Customers" VALUES(1,'John','Doe');
INSERT INTO "Customers" VALUES(2,'Jane','Smith');
INSERT INTO "Customers" VALUES(3,'Michael','Brown');
INSERT INTO "Customers" VALUES(4,'Emily','White');
INSERT INTO "Customers" VALUES(5,'David','Black');
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountNumber INTEGER,
    TransactionType TEXT, -- e.g., Subscription Payments, Transfer, etc.
    Amount REAL,
    Date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber));
INSERT INTO "Transactions" VALUES(1,1,'Subscription_Payments',150.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(2,1,'Transfer',200.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(3,2,'Subscription_Payments',130.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(4,2,'Transfer',180.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(5,3,'Subscription_Payments',140.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(6,3,'Transfer',220.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(7,4,'Subscription_Payments',120.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(8,4,'Transfer',190.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(9,5,'Subscription_Payments',110.0,'2024-01-30 18:53:44');
INSERT INTO "Transactions" VALUES(10,5,'Transfer',170.0,'2024-01-30 18:53:44');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',10);
INSERT INTO "sqlite_sequence" VALUES('Bank_Statements',5);
INSERT INTO "sqlite_sequence" VALUES('Bank_Statements_Transactions',14);
COMMIT;
