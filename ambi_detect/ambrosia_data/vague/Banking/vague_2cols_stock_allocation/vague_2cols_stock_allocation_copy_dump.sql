BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountNumber INTEGER PRIMARY KEY,
    Balance REAL ,
    InterestRate REAL DEFAULT 1,
    OpenDate DATE,
    CustomerID INTEGER,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,1000.0,2.0,'2023-01-01',1);
INSERT INTO "Accounts" VALUES(2,800.0,1.5,'2023-01-02',2);
INSERT INTO "Accounts" VALUES(3,700.0,1.0,'2023-01-03',3);
INSERT INTO "Accounts" VALUES(4,600.0,1.0,'2023-01-04',4);
INSERT INTO "Accounts" VALUES(5,900.0,2.5,'2023-01-05',5);
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Customers" VALUES(3,'Michael','Brown','michael.brown@example.com');
INSERT INTO "Customers" VALUES(4,'Emma','Wilson','emma.wilson@example.com');
INSERT INTO "Customers" VALUES(5,'David','Jackson','david.jackson@example.com');
CREATE TABLE Investments (
    InvestmentID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    StockSymbol TEXT,
    Quantity INTEGER ,
    PurchaseDate DATE,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber),
    FOREIGN KEY(StockSymbol) REFERENCES Stocks(StockSymbol)
);
INSERT INTO "Investments" VALUES(1,1,'AAPL',10,'2023-01-01');
INSERT INTO "Investments" VALUES(2,1,'GOOG',5,'2023-01-01');
INSERT INTO "Investments" VALUES(3,2,'AMZN',20,'2023-01-02');
INSERT INTO "Investments" VALUES(4,2,'MSFT',10,'2023-01-02');
INSERT INTO "Investments" VALUES(5,3,'TSLA',15,'2023-01-03');
CREATE TABLE "PortfolioPerformance" (
    PerformanceID INTEGER PRIMARY KEY,
    InvestmentID INTEGER,
    Stock_Allocation REAL ,
    Investment_Portfolios REAL,
    FOREIGN KEY(InvestmentID) REFERENCES Investments(InvestmentID)
);
INSERT INTO "PortfolioPerformance" VALUES(1,1,0.5,1000.0);
INSERT INTO "PortfolioPerformance" VALUES(2,2,0.3,4500.0);
INSERT INTO "PortfolioPerformance" VALUES(3,3,0.2,2000.0);
INSERT INTO "PortfolioPerformance" VALUES(4,4,0.4,8000.0);
INSERT INTO "PortfolioPerformance" VALUES(5,5,0.6,7200.0);
CREATE TABLE Stocks (
    StockSymbol TEXT PRIMARY KEY,
    CompanyName TEXT,
    CurrentPrice REAL
);
INSERT INTO "Stocks" VALUES('AAPL','Apple Inc.',150.0);
INSERT INTO "Stocks" VALUES('GOOG','Alphabet Inc.',2800.0);
INSERT INTO "Stocks" VALUES('AMZN','Amazon.com Inc.',130.0);
INSERT INTO "Stocks" VALUES('MSFT','Microsoft Corporation',300.0);
INSERT INTO "Stocks" VALUES('TSLA','Tesla Inc.',200.0);
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    Amount REAL,
    Type TEXT ,
    Date DATE,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber)
);
INSERT INTO "Transactions" VALUES(1,1,200.0,'Deposit','2023-01-01');
INSERT INTO "Transactions" VALUES(2,1,-50.0,'Withdrawal','2023-01-02');
INSERT INTO "Transactions" VALUES(3,2,300.0,'Deposit','2023-01-03');
INSERT INTO "Transactions" VALUES(4,2,-100.0,'Withdrawal','2023-01-04');
INSERT INTO "Transactions" VALUES(5,3,150.0,'Deposit','2023-01-05');
COMMIT;
