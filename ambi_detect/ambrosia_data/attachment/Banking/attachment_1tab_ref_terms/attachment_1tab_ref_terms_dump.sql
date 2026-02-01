BEGIN TRANSACTION;
CREATE TABLE AccountTypes (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    TypeName TEXT UNIQUE
);
INSERT INTO "AccountTypes" VALUES(1,'Checking');
INSERT INTO "AccountTypes" VALUES(2,'Savings');
INSERT INTO "AccountTypes" VALUES(3,'Money Market');
INSERT INTO "AccountTypes" VALUES(4,'CD');
INSERT INTO "AccountTypes" VALUES(5,'IRA');
CREATE TABLE BankAccounts (
    AccountNumber CHAR(10) PRIMARY KEY,
    CustomerID INT,
    OpenDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CurrentBalance REAL ,
    FOREIGN KEY (CustomerID) REFERENCES Customers(ID)
);
INSERT INTO "BankAccounts" VALUES('1834293214',1,'2021-06-01 10:00:00',10000.0);
INSERT INTO "BankAccounts" VALUES('2834293214',2,'2021-07-01 10:00:00',20000.0);
INSERT INTO "BankAccounts" VALUES('3834293214',3,'2021-08-01 10:00:00',30000.0);
INSERT INTO "BankAccounts" VALUES('4834293214',4,'2021-09-01 10:00:00',40000.0);
INSERT INTO "BankAccounts" VALUES('5834293214',5,'2021-10-01 10:00:00',50000.0);
CREATE TABLE Banks (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    HeadQuarters TEXT
);
INSERT INTO "Banks" VALUES(1,'Bank A','New York');
INSERT INTO "Banks" VALUES(2,'Bank B','London');
INSERT INTO "Banks" VALUES(3,'Bank C','Paris');
INSERT INTO "Banks" VALUES(4,'Bank D','Tokyo');
INSERT INTO "Banks" VALUES(5,'Bank E','Los Angeles');
CREATE TABLE Customers (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    Email TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','1976-05-15','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1980-08-20','jane.smith@example.com');
INSERT INTO "Customers" VALUES(3,'Peter','Black','1992-12-01','peter.black@example.com');
INSERT INTO "Customers" VALUES(4,'Alice','White','1985-03-17','alice.white@example.com');
INSERT INTO "Customers" VALUES(5,'Tom','Green','1995-07-05','tom.green@example.com');
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountNumber CHAR(10),
    Amount DECIMAL(15, 2),
    StartDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    EndDate TIMESTAMP,
    ForeignKeyAccountNumber CHAR(10) REFERENCES BankAccounts(AccountNumber),
    FOREIGN KEY (ForeignKeyAccountNumber) REFERENCES BankAccounts(AccountNumber)
);
INSERT INTO "Loans" VALUES(1,'1834293214',10000,'2021-06-01 10:00:00','2026-06-01 10:00:00','1834293214');
INSERT INTO "Loans" VALUES(2,'2834293214',15000,'2021-07-01 10:00:00','2027-07-01 10:00:00','2834293214');
INSERT INTO "Loans" VALUES(3,'3834293214',20000,'2021-08-01 10:00:00','2028-08-01 10:00:00','3834293214');
INSERT INTO "Loans" VALUES(4,'4834293214',25000,'2021-09-01 10:00:00','2029-09-01 10:00:00','4834293214');
INSERT INTO "Loans" VALUES(5,'5834293214',30000,'2021-10-01 10:00:00','2030-10-01 10:00:00','5834293214');
CREATE TABLE Terms (
    TermId INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    "Value" REAL,
    AccountTypeID INT,
    LOAN_ID INT,
    FOREIGN KEY (AccountTypeID) REFERENCES AccountTypes(ID),
    FOREIGN KEY (LOAN_ID) REFERENCES Loans(LoanID)
);
INSERT INTO "Terms" VALUES(1,'Monthly Interest Rate',0.05,1,1);
INSERT INTO "Terms" VALUES(2,'Monthly Interest Rate',0.06,2,2);
INSERT INTO "Terms" VALUES(3,'Annual Percentage Rate',0.03,1,1);
INSERT INTO "Terms" VALUES(4,'Annual Percentage Rate',0.07,2,2);
INSERT INTO "Terms" VALUES(5,'Some other value',0.05,3,3);
INSERT INTO "Terms" VALUES(6,'Another term',0.04,4,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('AccountTypes',5);
INSERT INTO "sqlite_sequence" VALUES('Banks',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Loans',5);
INSERT INTO "sqlite_sequence" VALUES('Terms',6);
COMMIT;
