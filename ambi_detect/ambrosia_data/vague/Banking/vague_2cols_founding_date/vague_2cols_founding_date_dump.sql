BEGIN TRANSACTION;
CREATE TABLE Accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerId INTEGER,
    Type TEXT ,
    Balance REAL,
    InterestRate REAL,
    OpenDate TEXT,
    FOREIGN KEY(CustomerId) REFERENCES Customers(id));
INSERT INTO "Accounts" VALUES(1,1,'Checking',1000.0,0.01,'2021-01-01');
INSERT INTO "Accounts" VALUES(2,2,'Savings',5000.0,0.02,'2021-02-01');
INSERT INTO "Accounts" VALUES(3,3,'Checking',2000.0,0.01,'2021-03-01');
INSERT INTO "Accounts" VALUES(4,4,'Savings',15000.0,0.02,'2021-04-01');
INSERT INTO "Accounts" VALUES(5,5,'Checking',3000.0,0.01,'2021-05-01');
CREATE TABLE Customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Address TEXT,
    PhoneNumber TEXT,
    EmailAddress TEXT,
    AccountId INTEGER,
    FOREIGN KEY(AccountId) REFERENCES Accounts(id));
INSERT INTO "Customers" VALUES(1,'John','Doe','1980-01-01','123 Main St','555-1234','john.doe@example.com',1);
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1975-02-01','456 Elm St','555-5678','jane.smith@example.com',2);
INSERT INTO "Customers" VALUES(3,'Alice','Johnson','1990-03-01','789 Oak St','555-9012','alice.johnson@example.com',3);
INSERT INTO "Customers" VALUES(4,'Michael','Williams','1985-04-01','135 Pine St','555-1357','michael.williams@example.com',4);
INSERT INTO "Customers" VALUES(5,'Emily','Taylor','1995-05-01','246 Maple Ave','555-2468','emily.taylor@example.com',5);
CREATE TABLE Financial_Institutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Founding_Year INTEGER,
    Founding_Decade TEXT,
    Total_Assets REAL,
    Total_Liabilities REAL,
    ROI_Percentage REAL);
INSERT INTO "Financial_Institutions" VALUES(1,'Bank A',1800,'19th century',2000000.0,1500000.0,0.25);
INSERT INTO "Financial_Institutions" VALUES(2,'Bank B',1963,'20th century',1000000.0,700000.0,0.15);
INSERT INTO "Financial_Institutions" VALUES(3,'Bank C',1988,'20th century',800000.0,500000.0,0.12);
INSERT INTO "Financial_Institutions" VALUES(4,'Bank D',2000,'21st century',600000.0,400000.0,0.1);
INSERT INTO "Financial_Institutions" VALUES(5,'Bank E',2010,'21st century',400000.0,300000.0,0.08);
CREATE TABLE Investments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    PurchaserId INTEGER,
    SecurityID INTEGER,
    Shares INTEGER,
    PricePerShare REAL,
    CurrentValue REAL,
    DatePurchased TEXT,
    FOREIGN KEY(PurchaserId) REFERENCES Customers(id),
    FOREIGN KEY(SecurityID) REFERENCES Securities(id));
INSERT INTO "Investments" VALUES(1,1,1,10,20.0,200.0,'2021-06-01');
INSERT INTO "Investments" VALUES(2,2,2,5,30.0,150.0,'2021-07-01');
INSERT INTO "Investments" VALUES(3,3,3,8,25.0,200.0,'2021-08-01');
INSERT INTO "Investments" VALUES(4,4,4,7,35.0,245.0,'2021-09-01');
INSERT INTO "Investments" VALUES(5,5,5,5,40.0,200.0,'2021-10-01');
CREATE TABLE Loans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    BorrowerId INTEGER,
    LenderId INTEGER,
    PrincipalAmount REAL,
    InterestRate REAL,
    TermYears INTEGER,
    MonthlyPayment REAL,
    StartDate TEXT,
    EndDate TEXT,
    FOREIGN KEY(BorrowerId) REFERENCES Customers(id),
    FOREIGN KEY(LenderId) REFERENCES Accounts(id));
INSERT INTO "Loans" VALUES(1,1,1,10000.0,0.05,5,211.23,'2021-06-01','2026-06-01');
INSERT INTO "Loans" VALUES(2,2,2,5000.0,0.04,3,156.25,'2021-07-01','2024-07-01');
INSERT INTO "Loans" VALUES(3,3,3,8000.0,0.06,4,228.0,'2021-08-01','2025-08-01');
INSERT INTO "Loans" VALUES(4,4,4,7000.0,0.05,5,190.42,'2021-09-01','2026-09-01');
INSERT INTO "Loans" VALUES(5,5,5,4000.0,0.045,3,140.0,'2021-10-01','2024-10-01');
CREATE TABLE Securities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Symbol TEXT,
    CompanyName TEXT,
    Sector TEXT,
    Industry TEXT);
INSERT INTO "Securities" VALUES(1,'AAPL','Apple Inc.','Technology','Consumer Electronics');
INSERT INTO "Securities" VALUES(2,'GOOG','Google LLC','Technology','Internet Services');
INSERT INTO "Securities" VALUES(3,'MSFT','Microsoft Corporation','Technology','Software');
INSERT INTO "Securities" VALUES(4,'AMZN','Amazon.com Inc.','Retail','E-commerce');
INSERT INTO "Securities" VALUES(5,'FB','Facebook, Inc.','Technology','Social Media');
CREATE TABLE Transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FromAccountId INTEGER,
    ToAccountId INTEGER,
    Amount REAL,
    TransactionType TEXT ,
    Date TEXT,
    FOREIGN KEY(FromAccountId) REFERENCES Accounts(id),
    FOREIGN KEY(ToAccountId) REFERENCES Accounts(id));
INSERT INTO "Transactions" VALUES(1,1,2,200.0,'Transfer','2021-01-15');
INSERT INTO "Transactions" VALUES(2,2,1,250.0,'Transfer','2021-01-15');
INSERT INTO "Transactions" VALUES(3,3,4,100.0,'Transfer','2021-03-15');
INSERT INTO "Transactions" VALUES(4,4,3,150.0,'Transfer','2021-03-15');
INSERT INTO "Transactions" VALUES(5,5,1,300.0,'Deposit','2021-05-15');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Financial_Institutions',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
INSERT INTO "sqlite_sequence" VALUES('Loans',5);
INSERT INTO "sqlite_sequence" VALUES('Investments',5);
INSERT INTO "sqlite_sequence" VALUES('Securities',5);
COMMIT;
