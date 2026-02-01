BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountNumber INTEGER PRIMARY KEY,
    Balance REAL,
    InterestRate FLOAT,
    OpenDate DATE,
    CustomerID INTEGER,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Accounts" VALUES(1,1000.0,0.05,'2021-01-01',1);
INSERT INTO "Accounts" VALUES(2,2000.0,0.03,'2021-02-01',2);
INSERT INTO "Accounts" VALUES(3,3000.0,0.04,'2021-03-01',3);
INSERT INTO "Accounts" VALUES(4,4000.0,0.02,'2021-04-01',4);
INSERT INTO "Accounts" VALUES(5,5000.0,0.06,'2021-05-01',5);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT);
INSERT INTO "Customers" VALUES(1,'John Smith','123 Main St');
INSERT INTO "Customers" VALUES(2,'Jane Doe','456 Oak Ave');
INSERT INTO "Customers" VALUES(3,'Alice Brown','789 Pine Blvd');
INSERT INTO "Customers" VALUES(4,'Bob White','101 Elm St');
INSERT INTO "Customers" VALUES(5,'Carol Green','202 Maple Dr');
CREATE TABLE InvestmentPortfolios(
    PortfolioID INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    OpenEndFunds TEXT,
    Month TEXT,
    TotalInvestments REAL,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "InvestmentPortfolios" VALUES(1,1,'Money Market Fund','January',10000.0);
INSERT INTO "InvestmentPortfolios" VALUES(2,2,'Money Market Fund','February',15000.0);
INSERT INTO "InvestmentPortfolios" VALUES(3,3,'Bond Fund','January',20000.0);
INSERT INTO "InvestmentPortfolios" VALUES(4,4,'Bond Fund','February',25000.0);
INSERT INTO "InvestmentPortfolios" VALUES(5,5,'Stock Fund','January',30000.0);
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY,
    Amount REAL,
    Date DATE,
    AccountNumber INTEGER,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber));
INSERT INTO "Transactions" VALUES(1,-500.0,'2021-01-15',1);
INSERT INTO "Transactions" VALUES(2,300.0,'2021-02-15',2);
INSERT INTO "Transactions" VALUES(3,-200.0,'2021-03-15',3);
INSERT INTO "Transactions" VALUES(4,400.0,'2021-04-15',4);
INSERT INTO "Transactions" VALUES(5,-1000.0,'2021-05-15',5);
COMMIT;
