BEGIN TRANSACTION;
CREATE TABLE Accounts (
    accountID INTEGER PRIMARY KEY,
    accountType TEXT,
    balance REAL ,
    customerID INTEGER,
    FOREIGN KEY(customerID) REFERENCES Customers(customerID));
INSERT INTO "Accounts" VALUES(1,'Checking',1000.0,1);
INSERT INTO "Accounts" VALUES(2,'Savings',500.0,2);
INSERT INTO "Accounts" VALUES(3,'Money Market',2000.0,3);
INSERT INTO "Accounts" VALUES(4,'CD',10000.0,4);
INSERT INTO "Accounts" VALUES(5,'IRA',5000.0,5);
CREATE TABLE "Banks" (
    bankID INTEGER PRIMARY KEY,
    bankName TEXT,
    branchNumber INTEGER,
    ceoID INTEGER,
    FOREIGN KEY(ceoID) REFERENCES "Leadership"(ceoID));
INSERT INTO "Banks" VALUES(1,'Bank A',100,1);
INSERT INTO "Banks" VALUES(2,'Bank B',200,2);
INSERT INTO "Banks" VALUES(3,'Bank C',300,3);
INSERT INTO "Banks" VALUES(4,'Bank D',400,4);
INSERT INTO "Banks" VALUES(5,'Bank E',500,5);
CREATE TABLE Customers (
    customerID INTEGER PRIMARY KEY,
    customerName TEXT,
    accountNumber INTEGER,
    bankID INTEGER,
    FOREIGN KEY(bankID) REFERENCES Banks(bankID));
INSERT INTO "Customers" VALUES(1,'Customer A',987654321,1);
INSERT INTO "Customers" VALUES(2,'Customer B',123456789,2);
INSERT INTO "Customers" VALUES(3,'Customer C',999999999,3);
INSERT INTO "Customers" VALUES(4,'Customer D',888888888,4);
INSERT INTO "Customers" VALUES(5,'Customer E',777777777,5);
CREATE TABLE "Leadership" (
    ceoID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Leadership" VALUES(1,'John','Doe');
INSERT INTO "Leadership" VALUES(2,'Jane','Smith');
INSERT INTO "Leadership" VALUES(3,'Michael','Johnson');
INSERT INTO "Leadership" VALUES(4,'Emily','Williams');
INSERT INTO "Leadership" VALUES(5,'David','Brown');
CREATE TABLE Transactions (
    transactionID INTEGER PRIMARY KEY,
    amount REAL,
    type TEXT,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    accountID INTEGER,
    FOREIGN KEY(accountID) REFERENCES Accounts(accountID));
INSERT INTO "Transactions" VALUES(1,500.0,'Deposit','2024-02-19 14:16:12',1);
INSERT INTO "Transactions" VALUES(2,-200.0,'Withdrawal','2024-02-19 14:16:12',2);
INSERT INTO "Transactions" VALUES(3,300.0,'Transfer','2024-02-19 14:16:12',3);
INSERT INTO "Transactions" VALUES(4,-400.0,'Withdrawal','2024-02-19 14:16:12',4);
INSERT INTO "Transactions" VALUES(5,500.0,'Deposit','2024-02-19 14:16:12',5);
COMMIT;
