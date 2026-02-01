BEGIN TRANSACTION;
CREATE TABLE AccountTypes(
    AccountTypeID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountTypeName TEXT);
INSERT INTO "AccountTypes" VALUES(1,'Checking Account');
INSERT INTO "AccountTypes" VALUES(2,'Savings Account');
INSERT INTO "AccountTypes" VALUES(3,'Credit Card');
INSERT INTO "AccountTypes" VALUES(4,'Mortgage Loan');
INSERT INTO "AccountTypes" VALUES(5,'Business Loan');
CREATE TABLE Accounts(
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    AccountTypeID INTEGER,
    OpenDate DATE,
    Balance REAL,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (AccountTypeID) REFERENCES AccountTypes(AccountTypeID));
INSERT INTO "Accounts" VALUES(1,1,1,'2020-01-01',1000.0);
INSERT INTO "Accounts" VALUES(2,2,2,'2021-01-01',5000.0);
INSERT INTO "Accounts" VALUES(3,3,3,'2022-01-01',-1000.0);
INSERT INTO "Accounts" VALUES(4,4,4,'2023-01-01',2000.0);
INSERT INTO "Accounts" VALUES(5,5,5,'2024-01-01',3000.0);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    DateOfBirth DATE,
    Digital_Solutions TEXT DEFAULT 'None',
    Sign_in_Options TEXT DEFAULT 'None', FullName TEXT);
INSERT INTO "Customers" VALUES(1,'2019-06-23','None','None','John Doe');
INSERT INTO "Customers" VALUES(2,'2018-07-12','None','None','Jane Smith');
INSERT INTO "Customers" VALUES(3,'2001-01-01','None','None','Alice Wonderland');
INSERT INTO "Customers" VALUES(4,'1980-12-31','None','None','Bob Builder');
INSERT INTO "Customers" VALUES(5,'2002-04-02','None','None','Charlie Banks');
INSERT INTO "Customers" VALUES(6,'1993-07-15','Mobile Banking App','Biometric Authentication','David Stone');
INSERT INTO "Customers" VALUES(7,'1994-08-16','Mobile Banking App','Password','Eva Green');
INSERT INTO "Customers" VALUES(8,'1985-09-17','Internet Banking Platform','Biometric Authentication','Frank Williams');
INSERT INTO "Customers" VALUES(9,'1906-12-09','Internet Banking Platform','PIN','Grace Hopper');
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    Amount REAL,
    Type TEXT,
    Date TIMESTAMP,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID));
INSERT INTO "Transactions" VALUES(1,1,500.0,'Deposit','2020-02-01');
INSERT INTO "Transactions" VALUES(2,2,1000.0,'Withdrawal','2021-03-01');
INSERT INTO "Transactions" VALUES(3,3,-200.0,'Transfer In','2022-04-01');
INSERT INTO "Transactions" VALUES(4,4,800.0,'Interest','2023-05-01');
INSERT INTO "Transactions" VALUES(5,5,1500.0,'Loan Payment','2024-06-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',9);
INSERT INTO "sqlite_sequence" VALUES('AccountTypes',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
COMMIT;
