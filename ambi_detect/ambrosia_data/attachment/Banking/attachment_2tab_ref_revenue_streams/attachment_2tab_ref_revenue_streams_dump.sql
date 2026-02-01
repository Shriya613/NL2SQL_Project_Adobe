BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Product TEXT,
    Balance REAL ,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,1,'Checking',1000.0);
INSERT INTO "Accounts" VALUES(2,2,'Savings',500.0);
INSERT INTO "Accounts" VALUES(3,3,'Money Market',2500.0);
INSERT INTO "Accounts" VALUES(4,4,'Premium Checking',5000.0);
INSERT INTO "Accounts" VALUES(5,5,'Certificates of Deposit',10000.0);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','2000-01-01');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1985-06-30');
INSERT INTO "Customers" VALUES(3,'Michael','Johnson','1992-07-15');
INSERT INTO "Customers" VALUES(4,'Emma','Williams','1997-08-20');
INSERT INTO "Customers" VALUES(5,'David','Martin','1990-09-25');
CREATE TABLE Fees(
    FeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    CardFee REAL ,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "Fees" VALUES(1,1,5.0);
INSERT INTO "Fees" VALUES(2,2,3.0);
INSERT INTO "Fees" VALUES(3,3,10.0);
INSERT INTO "Fees" VALUES(4,4,20.0);
INSERT INTO "Fees" VALUES(5,5,15.0);
CREATE TABLE InterestRates(
    RateID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    MonthlyInterestRate REAL ,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "InterestRates" VALUES(1,1,0.01);
INSERT INTO "InterestRates" VALUES(2,2,0.02);
INSERT INTO "InterestRates" VALUES(3,3,0.03);
INSERT INTO "InterestRates" VALUES(4,4,0.04);
INSERT INTO "InterestRates" VALUES(5,5,0.05);
CREATE TABLE Products(
    ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT,
    MonthlyInterestRate REAL ,
    CardFee REAL 
);
INSERT INTO "Products" VALUES(1,'Premium Credit Card','A high interest credit card.',0.02,5.0);
INSERT INTO "Products" VALUES(2,'Regular Credit Card','A standard credit card.',0.01,3.0);
INSERT INTO "Products" VALUES(3,'Business Loan','For business use only.',0.03,10.0);
INSERT INTO "Products" VALUES(4,'Personal Loan','Low interest personal loan.',0.04,20.0);
INSERT INTO "Products" VALUES(5,'Student Loan','For students only.',0.05,15.0);
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    Amount REAL ,
    Type TEXT, -- e.g., 'Deposit', 'Withdrawal'
    TransactionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "Transactions" VALUES(1,1,500.0,'Deposit','2024-03-13 20:52:19');
INSERT INTO "Transactions" VALUES(2,2,-100.0,'Withdrawal','2024-03-13 20:52:19');
INSERT INTO "Transactions" VALUES(3,3,200.0,'Deposit','2024-03-13 20:52:19');
INSERT INTO "Transactions" VALUES(4,4,-300.0,'Withdrawal','2024-03-13 20:52:19');
INSERT INTO "Transactions" VALUES(5,5,2500.0,'Deposit','2024-03-13 20:52:19');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
INSERT INTO "sqlite_sequence" VALUES('Fees',5);
INSERT INTO "sqlite_sequence" VALUES('InterestRates',5);
INSERT INTO "sqlite_sequence" VALUES('Products',5);
COMMIT;
