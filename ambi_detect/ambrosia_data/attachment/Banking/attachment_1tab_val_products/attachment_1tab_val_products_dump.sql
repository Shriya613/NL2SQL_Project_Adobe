BEGIN TRANSACTION;
CREATE TABLE AccountTypes(
    TypeID INTEGER PRIMARY KEY,
    TypeName VARCHAR(25));
INSERT INTO "AccountTypes" VALUES(1,'Checking');
INSERT INTO "AccountTypes" VALUES(2,'Savings');
CREATE TABLE Accounts(
    AccountNumber INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    AccountTypeID INTEGER,
    Balance REAL,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY(AccountTypeID) REFERENCES AccountTypes(TypeID));
INSERT INTO "Accounts" VALUES(1,1,1,1000.0);
INSERT INTO "Accounts" VALUES(2,2,2,500.0);
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT UNIQUE);
INSERT INTO "Customers" VALUES(1,'John','Doe','john@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane@example.com');
CREATE TABLE ExchangeRates(
    RateId INTEGER PRIMARY KEY,
    Currency VARCHAR(3));
INSERT INTO "ExchangeRates" VALUES(1,'USD');
INSERT INTO "ExchangeRates" VALUES(2,'EUR');
CREATE TABLE "Products"(
    ProductID INTEGER PRIMARY KEY,
    Name VARCHAR(25),
    Description TEXT,
    ExitFee REAL,
    Category VARCHAR(20));
INSERT INTO "Products" VALUES(1,'Exchange-Traded Product','This product tracks an index.',0.01,'ETP');
INSERT INTO "Products" VALUES(2,'Another Exchange-Traded Product','Another ETP tracking a different index.',0.02,'ETP');
INSERT INTO "Products" VALUES(3,'Unit Investment Trust','A trust investing in various assets.',0.01,'UIT');
INSERT INTO "Products" VALUES(4,'Another Unit Investment Trust','Another UIT with different asset allocation.',0.02,'UIT');
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    ProductID INTEGER,
    Quantity INTEGER,
    FOREIGN KEY(AccountNumber) REFERENCES Accounts(AccountNumber),
    FOREIGN KEY(ProductID) REFERENCES Products(ProductID));
INSERT INTO "Transactions" VALUES(1,1,1,10);
INSERT INTO "Transactions" VALUES(2,2,1,5);
INSERT INTO "Transactions" VALUES(3,1,2,20);
INSERT INTO "Transactions" VALUES(4,2,3,15);
COMMIT;
