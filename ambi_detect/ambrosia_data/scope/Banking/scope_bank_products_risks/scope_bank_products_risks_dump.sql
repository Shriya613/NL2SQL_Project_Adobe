BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountNumber CHAR(10),
    Balance REAL,
    OpenDate DATE);
INSERT INTO "Accounts" VALUES(1,'1234567890',10000.0,'2021-01-01');
INSERT INTO "Accounts" VALUES(2,'2345678901',5000.0,'2021-02-01');
INSERT INTO "Accounts" VALUES(3,'3456789012',15000.0,'2021-03-01');
INSERT INTO "Accounts" VALUES(4,'4567890123',20000.0,'2021-04-01');
INSERT INTO "Accounts" VALUES(5,'5678901234',25000.0,'2021-05-01');
CREATE TABLE Bank_Accounts(
    BankID INTEGER,
    AccountID INTEGER,
    FOREIGN KEY(BankID) REFERENCES Banks(BankID),
    FOREIGN KEY(AccountID) REFERENCES Accounts(AccountID));
INSERT INTO "Bank_Accounts" VALUES(1,1);
INSERT INTO "Bank_Accounts" VALUES(1,2);
INSERT INTO "Bank_Accounts" VALUES(2,3);
INSERT INTO "Bank_Accounts" VALUES(2,4);
INSERT INTO "Bank_Accounts" VALUES(3,5);
INSERT INTO "Bank_Accounts" VALUES(3,1);
INSERT INTO "Bank_Accounts" VALUES(4,2);
INSERT INTO "Bank_Accounts" VALUES(4,3);
INSERT INTO "Bank_Accounts" VALUES(5,4);
INSERT INTO "Bank_Accounts" VALUES(5,5);
CREATE TABLE Bank_Customers(
    BankID INTEGER,
    CustomerID INTEGER,
    FOREIGN KEY(BankID) REFERENCES Banks(BankID),
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Bank_Customers" VALUES(1,1);
INSERT INTO "Bank_Customers" VALUES(1,2);
INSERT INTO "Bank_Customers" VALUES(2,3);
INSERT INTO "Bank_Customers" VALUES(2,4);
INSERT INTO "Bank_Customers" VALUES(3,5);
INSERT INTO "Bank_Customers" VALUES(3,1);
INSERT INTO "Bank_Customers" VALUES(4,2);
INSERT INTO "Bank_Customers" VALUES(4,3);
INSERT INTO "Bank_Customers" VALUES(5,4);
INSERT INTO "Bank_Customers" VALUES(5,5);
CREATE TABLE Bank_Products(
    ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
    ProductName TEXT,
    InterestRate REAL);
INSERT INTO "Bank_Products" VALUES(1,'Checking Account',0.01);
INSERT INTO "Bank_Products" VALUES(2,'Savings Account',0.02);
INSERT INTO "Bank_Products" VALUES(3,'Fixed Deposit',0.03);
INSERT INTO "Bank_Products" VALUES(4,'Variable Deposit',0.04);
INSERT INTO "Bank_Products" VALUES(5,'Loan',0.05);
CREATE TABLE Bank_Products_Risks(
    ProductID INTEGER,
    RiskID INTEGER,
    FOREIGN KEY(ProductID) REFERENCES Bank_Products(ProductID),
    FOREIGN KEY(RiskID) REFERENCES Risks(RiskID));
INSERT INTO "Bank_Products_Risks" VALUES(1,1);
INSERT INTO "Bank_Products_Risks" VALUES(1,2);
INSERT INTO "Bank_Products_Risks" VALUES(2,1);
INSERT INTO "Bank_Products_Risks" VALUES(2,2);
INSERT INTO "Bank_Products_Risks" VALUES(3,1);
INSERT INTO "Bank_Products_Risks" VALUES(3,3);
INSERT INTO "Bank_Products_Risks" VALUES(4,1);
INSERT INTO "Bank_Products_Risks" VALUES(4,4);
INSERT INTO "Bank_Products_Risks" VALUES(5,1);
INSERT INTO "Bank_Products_Risks" VALUES(5,5);
CREATE TABLE Banks(
    BankID INTEGER PRIMARY KEY AUTOINCREMENT,
    BankName TEXT,
    Location TEXT);
INSERT INTO "Banks" VALUES(1,'First National','New York');
INSERT INTO "Banks" VALUES(2,'Global Bank','London');
INSERT INTO "Banks" VALUES(3,'Regional Bank','Chicago');
INSERT INTO "Banks" VALUES(4,'Local Bank','Boston');
INSERT INTO "Banks" VALUES(5,'Trust Company','San Francisco');
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT);
INSERT INTO "Customers" VALUES(1,'John Doe',32);
INSERT INTO "Customers" VALUES(2,'Jane Smith',28);
INSERT INTO "Customers" VALUES(3,'Mike Johnson',45);
INSERT INTO "Customers" VALUES(4,'Emily Davis',36);
INSERT INTO "Customers" VALUES(5,'David Brown',55);
CREATE TABLE Risks(
    RiskID INTEGER PRIMARY KEY AUTOINCREMENT,
    RiskName TEXT);
INSERT INTO "Risks" VALUES(1,'Inflation_Risk');
INSERT INTO "Risks" VALUES(2,'Interest_rate_risk');
INSERT INTO "Risks" VALUES(3,'Credit_risk');
INSERT INTO "Risks" VALUES(4,'Liquidity_risk');
INSERT INTO "Risks" VALUES(5,'Operational_risk');
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    Amount REAL,
    Date DATE,
    FOREIGN KEY(AccountID) REFERENCES Accounts(AccountID));
INSERT INTO "Transactions" VALUES(1,1,500.0,'2021-06-01');
INSERT INTO "Transactions" VALUES(2,1,-200.0,'2021-06-15');
INSERT INTO "Transactions" VALUES(3,2,300.0,'2021-06-01');
INSERT INTO "Transactions" VALUES(4,2,-100.0,'2021-06-15');
INSERT INTO "Transactions" VALUES(5,3,400.0,'2021-06-01');
INSERT INTO "Transactions" VALUES(6,3,-150.0,'2021-06-15');
INSERT INTO "Transactions" VALUES(7,4,500.0,'2021-06-01');
INSERT INTO "Transactions" VALUES(8,4,-200.0,'2021-06-15');
INSERT INTO "Transactions" VALUES(9,5,600.0,'2021-06-01');
INSERT INTO "Transactions" VALUES(10,5,-250.0,'2021-06-15');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Risks',5);
INSERT INTO "sqlite_sequence" VALUES('Banks',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',10);
INSERT INTO "sqlite_sequence" VALUES('Bank_Products',5);
COMMIT;
