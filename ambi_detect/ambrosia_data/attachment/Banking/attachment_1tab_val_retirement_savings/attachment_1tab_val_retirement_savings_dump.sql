BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClientID INTEGER,
    AccountType TEXT,
    Balance REAL ,
    FOREIGN KEY(ClientID) REFERENCES Clients(ClientID)
);
INSERT INTO "Accounts" VALUES(1,1,'Checking',2000.0);
INSERT INTO "Accounts" VALUES(2,1,'Savings',3000.0);
INSERT INTO "Accounts" VALUES(3,2,'Checking',5000.0);
INSERT INTO "Accounts" VALUES(4,2,'Savings',6000.0);
INSERT INTO "Accounts" VALUES(5,3,'Checking',7000.0);
INSERT INTO "Accounts" VALUES(6,3,'Savings',8000.0);
CREATE TABLE Clients(
    ClientID INTEGER PRIMARY KEY AUTOINCREMENT,
    Email TEXT UNIQUE
, FullName TEXT);
INSERT INTO "Clients" VALUES(1,'john@example.com','John Doe');
INSERT INTO "Clients" VALUES(2,'jane@example.com','Jane Smith');
INSERT INTO "Clients" VALUES(3,'alice@example.com','Alice Brown');
INSERT INTO "Clients" VALUES(4,'bob@example.com','Bob White');
INSERT INTO "Clients" VALUES(5,'carol@example.com','Carol Black');
CREATE TABLE Investments(
    InvestmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    Ticker TEXT,
    Shares INTEGER ,
    PricePerShare REAL ,
    TotalInvestment REAL ,
    FOREIGN KEY(AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "Investments" VALUES(1,1,'AAPL',50,150.0,7500.0);
INSERT INTO "Investments" VALUES(2,1,'GOOG',30,1200.0,36000.0);
INSERT INTO "Investments" VALUES(3,2,'AMZN',40,1100.0,44000.0);
INSERT INTO "Investments" VALUES(4,2,'MSFT',20,200.0,4000.0);
INSERT INTO "Investments" VALUES(5,3,'TSLA',30,300.0,9000.0);
CREATE TABLE Pensions(
    PensionID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClientID INTEGER,
    Retirement_Savings TEXT,
    Rate_of_return REAL,
    ContributionAmount REAL ,
    FOREIGN KEY(ClientID) REFERENCES Clients(ClientID)
);
INSERT INTO "Pensions" VALUES(1,1,'Pension Plan','5%',1000.0);
INSERT INTO "Pensions" VALUES(2,2,'Pension Plan','7%',1200.0);
INSERT INTO "Pensions" VALUES(3,3,'Brokerage Account','5%',900.0);
INSERT INTO "Pensions" VALUES(4,4,'Brokerage Account','8%',1100.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Clients',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',6);
INSERT INTO "sqlite_sequence" VALUES('Pensions',4);
INSERT INTO "sqlite_sequence" VALUES('Investments',5);
COMMIT;
