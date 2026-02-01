BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountNumber INTEGER PRIMARY KEY,
    Balance REAL ,
    InterestRate REAL,
    OpenDate DATE,
    CustomerID INTEGER,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID));
INSERT INTO "Accounts" VALUES(1001,5000.0,0.01,'2020-07-10',1);
INSERT INTO "Accounts" VALUES(1002,3000.0,0.02,'2020-08-15',2);
INSERT INTO "Accounts" VALUES(1003,8000.0,0.03,'2020-09-20',3);
CREATE TABLE "CardFees"(
    "Fee" REAL,
    FeeType TEXT,
    ProductID INTEGER,
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID));
INSERT INTO "CardFees" VALUES(100.0,'Annual Fee',1);
INSERT INTO "CardFees" VALUES(35.0,'Late Payment Fee',2);
CREATE TABLE CustomerAccounts(
    CustomerAccountID INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    AccountNumber INTEGER,
    OpenDate DATE,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber));
INSERT INTO "CustomerAccounts" VALUES(101,1,1001,'2020-07-10');
INSERT INTO "CustomerAccounts" VALUES(102,2,1002,'2020-08-15');
INSERT INTO "CustomerAccounts" VALUES(103,3,1003,'2020-09-20');
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    EmailAddress TEXT UNIQUE);
INSERT INTO "Customers" VALUES(1,'John','Doe','2000-01-01','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1985-06-15','jane.smith@example.com');
INSERT INTO "Customers" VALUES(3,'Alice','Brown','1990-10-20','alice.brown@example.com');
CREATE TABLE "MonthlyInterestRates"(
    Rate REAL,
    RateType TEXT,
    EffectiveDate DATE,
    ExpirationDate DATE,
    ProductID INTEGER,
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID));
INSERT INTO "MonthlyInterestRates" VALUES(0.01,'Fixed Rate','2021-01-01','2021-12-31',1);
INSERT INTO "MonthlyInterestRates" VALUES(0.03,'Variable Rate','2022-01-01','2022-12-31',3);
CREATE TABLE Products(
    ProductID INTEGER PRIMARY KEY,
    Name TEXT,
    Type TEXT, -- e.g., Savings account, Loan, etc.
    Description TEXT);
INSERT INTO "Products" VALUES(1,'Premium Credit Card','Credit Card','A high-interest credit card with low fees.');
INSERT INTO "Products" VALUES(2,'Basic Checking Account','Checking Account','A basic checking account with no monthly fee.');
INSERT INTO "Products" VALUES(3,'High Yield Savings Account','Savings Account','Earn a higher interest rate on your savings balance.');
COMMIT;
