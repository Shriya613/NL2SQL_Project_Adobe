BEGIN TRANSACTION;
CREATE TABLE Accounts(
    accountID INTEGER PRIMARY KEY AUTOINCREMENT,
    customerID INTEGER,
    balance REAL ,
    FOREIGN KEY(customerID) REFERENCES Customers(customerID));
INSERT INTO "Accounts" VALUES(1,1,1000.0);
INSERT INTO "Accounts" VALUES(2,2,-500.0);
INSERT INTO "Accounts" VALUES(3,3,2000.0);
INSERT INTO "Accounts" VALUES(4,4,-300.0);
INSERT INTO "Accounts" VALUES(5,5,500.0);
CREATE TABLE Customers(
    customerID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE,
    address TEXT);
INSERT INTO "Customers" VALUES(1,'John','Doe','1987-06-12','New York');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1990-05-15','Los Angeles');
INSERT INTO "Customers" VALUES(3,'Tom','Nelson','1975-08-18','Chicago');
INSERT INTO "Customers" VALUES(4,'Alice','Williams','1985-09-20','Houston');
INSERT INTO "Customers" VALUES(5,'Bob','Jackson','1992-11-22','Seattle');
CREATE TABLE Mortgages(
    mortgageID INTEGER PRIMARY KEY AUTOINCREMENT,
    accountID INTEGER,
    loanType TEXT ,
    principalAmount REAL ,
    annualInterestRate DECIMAL(5,2) ,
    repaymentPeriod TEXT,
    FOREIGN KEY(accountID) REFERENCES Accounts(accountID));
INSERT INTO "Mortgages" VALUES(1,1,'Mortgage',100000.0,0.05,'30');
INSERT INTO "Mortgages" VALUES(2,2,'Mortgage',200000.0,0.06,'25');
INSERT INTO "Mortgages" VALUES(3,3,'Home Equity Loan',50000.0,0.04,'20');
INSERT INTO "Mortgages" VALUES(4,4,'Home Equity Loan',30000.0,0.05,'30');
INSERT INTO "Mortgages" VALUES(5,5,'Mortgage',150000.0,0.07,'15');
CREATE TABLE SavingsAccounts(
    savingsID INTEGER PRIMARY KEY AUTOINCREMENT,
    accountID INTEGER,
    interestRate DECIMAL(5,2) ,
    FOREIGN KEY(accountID) REFERENCES Accounts(accountID));
INSERT INTO "SavingsAccounts" VALUES(1,1,0.03);
INSERT INTO "SavingsAccounts" VALUES(2,2,0.04);
INSERT INTO "SavingsAccounts" VALUES(3,3,0.02);
INSERT INTO "SavingsAccounts" VALUES(4,4,0.05);
INSERT INTO "SavingsAccounts" VALUES(5,5,0.03);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Mortgages',5);
INSERT INTO "sqlite_sequence" VALUES('SavingsAccounts',5);
COMMIT;
