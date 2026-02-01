BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountNumber INTEGER PRIMARY KEY,
    Balance REAL ,
    CustomerID INTEGER,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,1234.67,1);
INSERT INTO "Accounts" VALUES(2,-543.98,2);
INSERT INTO "Accounts" VALUES(3,7654.32,3);
INSERT INTO "Accounts" VALUES(4,-234.56,4);
INSERT INTO "Accounts" VALUES(5,12345.67,5);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane@example.com');
INSERT INTO "Customers" VALUES(3,'Bob','Building','bob@example.com');
INSERT INTO "Customers" VALUES(4,'Alice','Adams','alice@example.com');
INSERT INTO "Customers" VALUES(5,'Tom','Turner','tom@example.com');
CREATE TABLE InvestmentCompanies(
    CompanyID INTEGER PRIMARY KEY,
    Name TEXT,
    Type TEXT,
    FundingAmount REAL
);
INSERT INTO "InvestmentCompanies" VALUES(1,'Pinnacle Growth Equity','Growth Equity Firm',5000000.0);
INSERT INTO "InvestmentCompanies" VALUES(2,'Summit Equity Partners','Growth Equity Firm',6000000.0);
INSERT INTO "InvestmentCompanies" VALUES(3,'Innovate Venture Capital','Venture Capital Trust',5000000.0);
INSERT INTO "InvestmentCompanies" VALUES(4,'Frontier VC Group','Venture Capital Trust',4000000.0);
INSERT INTO "InvestmentCompanies" VALUES(5,'Elite Private Equity','Private Equity Firm',7000000.0);
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY,
    Date DATETIME DEFAULT CURRENT_TIMESTAMP,
    Amount REAL,
    AccountNumber INTEGER,
    CompanyID INTEGER,
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber),
    FOREIGN KEY (CompanyID) REFERENCES InvestmentCompanies(CompanyID)
);
INSERT INTO "Transactions" VALUES(1,'2024-03-07 22:58:27',1000.0,1,1);
INSERT INTO "Transactions" VALUES(2,'2024-03-07 22:58:27',-500.0,2,2);
INSERT INTO "Transactions" VALUES(3,'2024-03-07 22:58:27',700.0,3,3);
INSERT INTO "Transactions" VALUES(4,'2024-03-07 22:58:27',-300.0,4,4);
INSERT INTO "Transactions" VALUES(5,'2024-03-07 22:58:27',1500.0,5,5);
COMMIT;
