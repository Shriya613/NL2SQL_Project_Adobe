BEGIN TRANSACTION;
CREATE TABLE Accounts(
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClientID INTEGER,
    Balance REAL ,
    InterestRate REAL ,
    FOREIGN KEY(ClientID) REFERENCES Clients(ClientID)
);
INSERT INTO "Accounts" VALUES(1,1,5000.0,0.05);
INSERT INTO "Accounts" VALUES(2,2,10000.0,0.03);
CREATE TABLE Clients(
    ClientID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT,
    PhoneNumber VARCHAR(20)
);
INSERT INTO "Clients" VALUES(1,'John','Smith','johnsmith@email.com','+123456789');
INSERT INTO "Clients" VALUES(2,'Jane','Doe','janedoe@email.com','+987654321');
CREATE TABLE Loans(
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    ClientID INTEGER,
    Principal AMOUNT ,
    InterestRate REAL ,
    TermYears INTEGER ,
    MonthlyPayment AMOUNT ,
    FOREIGN KEY(ClientID) REFERENCES Clients(ClientID)
);
INSERT INTO "Loans" VALUES(1,1,15000,0.06,5,400);
INSERT INTO "Loans" VALUES(2,2,20000,0.05,10,500);
CREATE TABLE Structured_Financial_Products(
    ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Type TEXT,
    Price REAL ,
    OtherValues TEXT
);
INSERT INTO "Structured_Financial_Products" VALUES(1,'CDO1','Collateralized Debt Obligation',1000.0,'Details about CDO1');
INSERT INTO "Structured_Financial_Products" VALUES(2,'CDO2','Collateralized Debt Obligation',1200.0,'Details about CDO2');
INSERT INTO "Structured_Financial_Products" VALUES(3,'CMO1','Collateralized Mortgage Obligation',1000.0,'Details about CMO1');
INSERT INTO "Structured_Financial_Products" VALUES(4,'CMO2','Collateralized Mortgage Obligation',1500.0,'Details about CMO2');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Accounts',2);
INSERT INTO "sqlite_sequence" VALUES('Loans',2);
INSERT INTO "sqlite_sequence" VALUES('Structured_Financial_Products',4);
COMMIT;
