BEGIN TRANSACTION;
CREATE TABLE Accounts (
    accountNumber INTEGER PRIMARY KEY AUTOINCREMENT,
    balance REAL ,
    interestRate REAL ,
    customerId INTEGER,
    FOREIGN KEY(customerId) REFERENCES Customers(customerId)
);
INSERT INTO "Accounts" VALUES(1,1000.0,0.02,1);
INSERT INTO "Accounts" VALUES(2,5000.0,0.03,2);
INSERT INTO "Accounts" VALUES(3,8000.0,0.01,3);
INSERT INTO "Accounts" VALUES(4,2000.0,0.04,4);
INSERT INTO "Accounts" VALUES(5,3000.0,0.05,5);
CREATE TABLE Banks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    Ceo TEXT,
    Management_Team TEXT
);
INSERT INTO "Banks" VALUES(1,'Bank A','New York','John Doe','Jane Smith, Mike Johnson');
INSERT INTO "Banks" VALUES(2,'Bank B','Los Angeles','Emma Jones','David Brown, Sarah Lee');
INSERT INTO "Banks" VALUES(3,'Bank C','Chicago','Tom Clark','Lisa Miller, Kevin Wright');
INSERT INTO "Banks" VALUES(4,'Bank D','Miami','Rachel Taylor','Paul King, Susan White');
INSERT INTO "Banks" VALUES(5,'Bank E','Dallas','Steve Martin','Kathy Davis, Mark Johnson');
CREATE TABLE Customers (
    customerId INTEGER PRIMARY KEY AUTOINCREMENT,
    dob DATE,
    phoneNumber TEXT,
    emailAddress TEXT,
    bankID INTEGER, fullname TEXT,
    FOREIGN KEY(bankID) REFERENCES Banks(id)
);
INSERT INTO "Customers" VALUES(1,'1980-07-23','555-1234','alice@example.com',1,'Alice Smith');
INSERT INTO "Customers" VALUES(2,'1985-11-10','555-5678','bob@example.com',1,'Bob Johnson');
INSERT INTO "Customers" VALUES(3,'1990-03-15','555-9012','carol@example.com',2,'Carol Williams');
INSERT INTO "Customers" VALUES(4,'1988-06-27','555-1357','daniel@example.com',3,'Daniel Davis');
INSERT INTO "Customers" VALUES(5,'1992-10-12','555-2468','eva@example.com',4,'Eva Martin');
CREATE TABLE Loans (
    loanId INTEGER PRIMARY KEY AUTOINCREMENT,
    principalAmount REAL ,
    interestRate REAL ,
    termMonths INTEGER ,
    monthlyPayment REAL ,
    customerId INTEGER,
    FOREIGN KEY(customerId) REFERENCES Customers(customerId)
);
INSERT INTO "Loans" VALUES(1,20000.0,0.05,36,650.0,1);
INSERT INTO "Loans" VALUES(2,15000.0,0.04,36,500.0,2);
INSERT INTO "Loans" VALUES(3,30000.0,0.06,48,800.0,3);
INSERT INTO "Loans" VALUES(4,40000.0,0.05,60,850.0,4);
INSERT INTO "Loans" VALUES(5,25000.0,0.04,48,550.0,5);
CREATE TABLE Transactions (
    transactionId INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL ,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    sourceAccountNumber INTEGER,
    destinationAccountNumber INTEGER,
    FOREIGN KEY(sourceAccountNumber) REFERENCES Accounts(accountNumber),
    FOREIGN KEY(destinationAccountNumber) REFERENCES Accounts(accountNumber)
);
INSERT INTO "Transactions" VALUES(1,-200.0,NULL,'Transfer to friend',1,2);
INSERT INTO "Transactions" VALUES(2,150.0,NULL,'Online shopping',3,4);
INSERT INTO "Transactions" VALUES(3,50.0,NULL,'ATM withdrawal',5,5);
INSERT INTO "Transactions" VALUES(4,300.0,NULL,'Salary credit',6,6);
INSERT INTO "Transactions" VALUES(5,200.0,NULL,'Bill payment',7,7);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Banks',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
INSERT INTO "sqlite_sequence" VALUES('Loans',5);
COMMIT;
