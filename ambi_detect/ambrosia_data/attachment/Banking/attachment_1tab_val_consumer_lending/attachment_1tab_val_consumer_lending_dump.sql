BEGIN TRANSACTION;
CREATE TABLE Account(
    AccountNumber INTEGER PRIMARY KEY AUTOINCREMENT,
    Balance REAL,
    InterestRate REAL,
    ConsumerLending TEXT,
    ForeignKeyCustomerID INTEGER,
    FOREIGN KEY(ForeignKeyCustomerID) REFERENCES Customer(CustomerID));
INSERT INTO "Account" VALUES(1,1000.0,15.0,'Credit Card',NULL);
INSERT INTO "Account" VALUES(2,2000.0,10.0,'Credit Card',NULL);
INSERT INTO "Account" VALUES(3,3000.0,15.0,'Personal Loan',NULL);
INSERT INTO "Account" VALUES(4,4000.0,12.0,'Personal Loan',NULL);
CREATE TABLE AccountType(
    TypeID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountTypeName TEXT UNIQUE);
INSERT INTO "AccountType" VALUES(1,'Checking');
INSERT INTO "AccountType" VALUES(2,'Savings');
CREATE TABLE Customer(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE);
INSERT INTO "Customer" VALUES(1,'John','Doe','2000-01-01');
INSERT INTO "Customer" VALUES(2,'Jane','Smith','1987-06-15');
CREATE TABLE Transactions(
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Amount REAL,
    TypeID INTEGER,
    AccountNumber INTEGER,
    ForeignKeyAccountNumber INTEGER,
    FOREIGN KEY(ForeignKeyAccountNumber) REFERENCES Account(AccountNumber),
    FOREIGN KEY(TypeID) REFERENCES AccountType(TypeID));
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customer',2);
INSERT INTO "sqlite_sequence" VALUES('AccountType',2);
INSERT INTO "sqlite_sequence" VALUES('Account',4);
COMMIT;
