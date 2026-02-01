BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    Balance REAL ,
    InterestRate FLOAT ,
    DateOpened DATE,
    CustomerID INTEGER,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,1000.0,2.5,'2020-01-01',1);
INSERT INTO "Accounts" VALUES(2,2000.0,3.0,'2020-01-02',2);
INSERT INTO "Accounts" VALUES(3,3000.0,2.0,'2020-01-03',3);
INSERT INTO "Accounts" VALUES(4,4000.0,2.5,'2020-01-04',4);
INSERT INTO "Accounts" VALUES(5,5000.0,3.0,'2020-01-05',5);
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john@example.com','+123456789');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane@example.com','+987654321');
INSERT INTO "Customers" VALUES(3,'Bob','Robertson','bob@example.com','+0987654321');
INSERT INTO "Customers" VALUES(4,'Alice','Wonderland','alice@example.com','+135792468');
INSERT INTO "Customers" VALUES(5,'Charlie','Brown','charlie@example.com','+246801753');
CREATE TABLE Insurance_Products (
    ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
    PolicyHolderID INTEGER,
    ProductType TEXT ,
    Payout REAL ,
    StartDate DATE,
    EndDate DATE,
    FOREIGN KEY (PolicyHolderID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Insurance_Products" VALUES(1,1,'Life Insurance',100000.0,'2020-01-01','2030-01-01');
INSERT INTO "Insurance_Products" VALUES(2,2,'Life Insurance',50000.0,'2020-01-02','2030-01-02');
INSERT INTO "Insurance_Products" VALUES(3,3,'Annuity',100000.0,'2020-01-03','2030-01-03');
INSERT INTO "Insurance_Products" VALUES(4,4,'Annuity',50000.0,'2020-01-04','2030-01-04');
INSERT INTO "Insurance_Products" VALUES(5,5,'Health Insurance',80000.0,'2020-01-05','2030-01-05');
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    Principal REAL ,
    InterestRate FLOAT ,
    TermYears INTEGER ,
    MonthlyPayment REAL ,
    ApprovalDate DATE,
    AccountID INTEGER,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "Loans" VALUES(1,10000.0,5.0,30,450.0,NULL,1);
INSERT INTO "Loans" VALUES(2,15000.0,4.5,25,600.0,NULL,2);
INSERT INTO "Loans" VALUES(3,20000.0,5.5,30,500.0,NULL,3);
INSERT INTO "Loans" VALUES(4,25000.0,5.0,20,750.0,NULL,4);
INSERT INTO "Loans" VALUES(5,30000.0,4.5,25,650.0,NULL,5);
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Amount REAL ,
    Type TEXT ,
    Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    AccountID INTEGER,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "Transactions" VALUES(1,-500.0,'Withdrawal','2020-01-01',1);
INSERT INTO "Transactions" VALUES(2,200.0,'Deposit','2020-01-02',2);
INSERT INTO "Transactions" VALUES(3,300.0,'Transfer','2020-01-03',3);
INSERT INTO "Transactions" VALUES(4,-200.0,'Withdrawal','2020-01-04',4);
INSERT INTO "Transactions" VALUES(5,500.0,'Deposit','2020-01-05',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Insurance_Products',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
INSERT INTO "sqlite_sequence" VALUES('Loans',5);
COMMIT;
