BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountNumber INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    Balance REAL DEFAULT 0,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
CREATE TABLE Currencies(
    CurrencyID INTEGER PRIMARY KEY,
    Name TEXT,
    Symbol TEXT
);
INSERT INTO "Currencies" VALUES(1,'USD','$');
INSERT INTO "Currencies" VALUES(2,'EUR','€');
INSERT INTO "Currencies" VALUES(3,'GBP','£');
INSERT INTO "Currencies" VALUES(4,'AUD','$');
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY,
    Organization TEXT,
    Year_of_establishment INT
, FullName TEXT);
INSERT INTO "Customers" VALUES(1,'Currency Exchange Service',2000,'John Doe');
INSERT INTO "Customers" VALUES(2,'Currency Exchange Service',1998,'Jane Smith');
INSERT INTO "Customers" VALUES(3,'Commodity Trading Firm',2000,'Michael Brown');
INSERT INTO "Customers" VALUES(4,'Commodity Trading Firm',1996,'Emma White');
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY,
    AccountNumber INTEGER,
    Amount REAL,
    CurrencyID INTEGER,
    Date DATETIME,
    FOREIGN KEY (AccountNumber) REFERENCES Accounts(AccountNumber),
    FOREIGN KEY (CurrencyID) REFERENCES Currencies(CurrencyID)
);
COMMIT;
