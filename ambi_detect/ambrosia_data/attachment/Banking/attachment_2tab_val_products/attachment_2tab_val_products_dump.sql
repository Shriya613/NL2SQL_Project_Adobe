BEGIN TRANSACTION;
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY,
    CustomerName TEXT,
    EmailAddress TEXT,
    PhoneNumber VARCHAR(10),
    AccountBalance REAL
);
INSERT INTO "Customers" VALUES(1,'John Doe','johndoe@example.com','1234567890',5000.0);
INSERT INTO "Customers" VALUES(2,'Jane Smith','janesmith@example.com','0987654321',10000.0);
INSERT INTO "Customers" VALUES(3,'Alice Johnson','alicejohnson@example.com','5555555555',15000.0);
INSERT INTO "Customers" VALUES(4,'Bob Brown','bobbrown@example.com','5555555556',20000.0);
INSERT INTO "Customers" VALUES(5,'Charlie Green','charliegreen@example.com','5555555557',25000.0);
CREATE TABLE Exchange_traded_Products(
    id INTEGER PRIMARY KEY,
    ProductName TEXT,
    TickerSymbol TEXT,
    AssetClass TEXT,
    TotalAssets REAL,
    ExitFee DECIMAL(5,2)
);
INSERT INTO "Exchange_traded_Products" VALUES(1,'AAPL','AAPL','Equity',239.67,0.01);
INSERT INTO "Exchange_traded_Products" VALUES(2,'MSFT','MSFT','Equity',286.21,0.015);
INSERT INTO "Exchange_traded_Products" VALUES(3,'GOOG','GOOG','Equity',262.33,0.01);
INSERT INTO "Exchange_traded_Products" VALUES(4,'AMZN','AMZN','Equity',356.33,0.01);
INSERT INTO "Exchange_traded_Products" VALUES(5,'TSLA','TSLA','Equity',109.86,0.01);
CREATE TABLE Orders(
    OrderID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerId INTEGER,
    ProductType TEXT,
    Quantity INTEGER ,
    Price PERSON,
    DateOrdered DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (CustomerId) REFERENCES Customers(id)
);
INSERT INTO "Orders" VALUES(1,1,'Exchange_traded_product',10,100,'2022-01-01 12:00:00');
INSERT INTO "Orders" VALUES(2,2,'Exchange_traded_product',5,120,'2022-01-01 12:01:00');
INSERT INTO "Orders" VALUES(3,3,'Exchange_traded_product',10,110,'2022-01-01 12:02:00');
INSERT INTO "Orders" VALUES(4,4,'Exchange_traded_product',15,130,'2022-01-01 12:03:00');
INSERT INTO "Orders" VALUES(5,5,'Exchange_traded_product',20,140,'2022-01-01 12:04:00');
CREATE TABLE Transactions(
    TransactionId INTEGER PRIMARY KEY AUTOINCREMENT,
    OrderId INTEGER,
    Amount REAL,
    Fees REAL,
    PaymentMethod TEXT,
    Status TEXT ,
    ProcessDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (OrderId) REFERENCES Orders(OrderID)
);
INSERT INTO "Transactions" VALUES(1,1,1000.0,20.0,'Debit Card','Completed','2022-01-01 12:00:00');
INSERT INTO "Transactions" VALUES(2,2,600.0,12.0,'PayPal','Completed','2022-01-01 12:01:00');
INSERT INTO "Transactions" VALUES(3,3,1100.0,22.0,'Bank Transfer','Completed','2022-01-01 12:02:00');
INSERT INTO "Transactions" VALUES(4,4,1300.0,24.0,'Check','Completed','2022-01-01 12:03:00');
INSERT INTO "Transactions" VALUES(5,5,1400.0,26.0,'Wire Transfer','Completed','2022-01-01 12:04:00');
CREATE TABLE Unit_investment_Trusts(
    id INTEGER PRIMARY KEY,
    TrustName TEXT,
    TickerSymbol TEXT,
    NetAssetValue REAL,
    NumberOfUnits INTEGER,
    ExitFee DECIMAL(5,2)
);
INSERT INTO "Unit_investment_Trusts" VALUES(1,'ABCD','ABCD',10.0,100000,0.01);
INSERT INTO "Unit_investment_Trusts" VALUES(2,'EFGH','EFGH',12.5,80000,0.015);
INSERT INTO "Unit_investment_Trusts" VALUES(3,'IJKL','IJKL',8.2,50000,0.01);
INSERT INTO "Unit_investment_Trusts" VALUES(4,'MNOP','MNOP',7.6,60000,0.015);
INSERT INTO "Unit_investment_Trusts" VALUES(5,'QRST','QRST',9.8,75000,0.01);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Orders',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
COMMIT;
