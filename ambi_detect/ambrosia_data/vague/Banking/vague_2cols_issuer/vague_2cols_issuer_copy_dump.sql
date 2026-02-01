BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    AccountType TEXT,
    Balance REAL ,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,1,'Checking',500.0);
INSERT INTO "Accounts" VALUES(2,2,'Savings',1000.0);
CREATE TABLE Bank (
    BankName TEXT PRIMARY KEY,
    HeadquarterLocation TEXT,
    FoundedYear INTEGER,
    TotalAssets REAL,
    NumberOfBranches INTEGER,
    WebsiteURL TEXT UNIQUE
);
INSERT INTO "Bank" VALUES('Example Bank','New York',2000,1000000.0,50,'https://www.examplebank.com/');
CREATE TABLE "BankingProducts" (
    ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
    Issuer TEXT,
    ProductName TEXT,
    Description TEXT,
    AnnualInterestRate REAL,
    MinimumDepositAmount REAL,
    MaximumDepositAmount REAL,
    FOREIGN KEY (Issuer) REFERENCES Bank(BankName)
);
INSERT INTO "BankingProducts" VALUES(1,'Example Bank','High-Yield Savings','A high interest rate savings account with no monthly fees.',1.0,1000.0,NULL);
INSERT INTO "BankingProducts" VALUES(2,'Example Bank','CD Special','Certificate of deposit offering a fixed annual percentage yield.',2.5,1000.0,1000000.0);
CREATE TABLE Branch (
    BranchID INTEGER PRIMARY KEY AUTOINCREMENT,
    BankName TEXT,
    BranchName TEXT,
    Address TEXT,
    City TEXT,
    State TEXT,
    ZipCode TEXT,
    FOREIGN KEY (BankName) REFERENCES Bank(BankName)
);
INSERT INTO "Branch" VALUES(1,'Example Bank','Main Branch','123 Main St','New York','NY','10001');
INSERT INTO "Branch" VALUES(2,'Example Bank','East Side Branch','456 East Ave','New York','NY','10010');
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber VARCHAR(15) UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com','+1234567890');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com','+0987654321');
CREATE TABLE Deposits (
    DepositID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    Amount REAL,
    Date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "Deposits" VALUES(1,1,500.0,'2024-02-19 14:31:33');
INSERT INTO "Deposits" VALUES(2,2,1000.0,'2024-02-19 14:31:33');
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Amount REAL,
    InterestRate REAL,
    TermYears INTEGER,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Loans" VALUES(1,1,15000.0,6.0,5);
INSERT INTO "Loans" VALUES(2,2,10000.0,5.5,3);
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER,
    Type TEXT, -- 'Deposit' or 'Withdrawal'
    Amount REAL,
    Date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID)
);
INSERT INTO "Transactions" VALUES(1,1,'Deposit',500.0,'2024-02-19 14:31:33');
INSERT INTO "Transactions" VALUES(2,2,'Withdrawal',200.0,'2024-02-19 14:31:33');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',2);
INSERT INTO "sqlite_sequence" VALUES('Accounts',2);
INSERT INTO "sqlite_sequence" VALUES('Loans',2);
INSERT INTO "sqlite_sequence" VALUES('Deposits',2);
INSERT INTO "sqlite_sequence" VALUES('Transactions',2);
INSERT INTO "sqlite_sequence" VALUES('Branch',2);
INSERT INTO "sqlite_sequence" VALUES('BankingProducts',2);
COMMIT;
