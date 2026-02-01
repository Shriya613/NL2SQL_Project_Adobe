BEGIN TRANSACTION;
CREATE TABLE Accounts (
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    Number CHAR(9),
    OpenDate DATE,
    Balance REAL ,
    CustomerID INTEGER,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Accounts" VALUES(1,'1234567890','2021-01-01',1000.0,1);
INSERT INTO "Accounts" VALUES(2,'9876543210','2021-02-02',2000.0,2);
INSERT INTO "Accounts" VALUES(3,'5432109876','2021-03-03',3000.0,3);
INSERT INTO "Accounts" VALUES(4,'6543217890','2021-04-04',4000.0,4);
INSERT INTO "Accounts" VALUES(5,'7890123456','2021-05-05',5000.0,5);
CREATE TABLE Banks (
    BankID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Banks" VALUES(1,'ABC Bank');
INSERT INTO "Banks" VALUES(2,'XYZ Bank');
INSERT INTO "Banks" VALUES(3,'PQR Bank');
INSERT INTO "Banks" VALUES(4,'LMN Bank');
INSERT INTO "Banks" VALUES(5,'JKL Bank');
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    DateOfBirth DATE,
    PhoneNumber VARCHAR(10),
    EmailAddress VARCHAR(255),
    BankID INTEGER,
    FOREIGN KEY (BankID) REFERENCES Banks(BankID)
);
INSERT INTO "Customers" VALUES(1,'John','Doe','1980-01-01','1234567890','john@example.com',1);
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1990-02-02','0987654321','jane@example.com',2);
INSERT INTO "Customers" VALUES(3,'Alice','Johnson','1995-03-03','5551234567','alice@example.com',3);
INSERT INTO "Customers" VALUES(4,'Bob','Williams','2000-04-04','5555555555','bob@example.com',4);
INSERT INTO "Customers" VALUES(5,'Charlie','Miller','2005-05-05','5555555555','charlie@example.com',5);
CREATE TABLE Transaction_Details (
    DetailsID INTEGER PRIMARY KEY AUTOINCREMENT,
    TransactionID INTEGER,
    AdditionalInfo TEXT,
    FOREIGN KEY (TransactionID) REFERENCES Transactions(TransactionID)
);
INSERT INTO "Transaction_Details" VALUES(1,1,'Additional info for transaction 1');
INSERT INTO "Transaction_Details" VALUES(2,2,'Additional info for transaction 2');
INSERT INTO "Transaction_Details" VALUES(3,3,'Additional info for transaction 3');
INSERT INTO "Transaction_Details" VALUES(4,4,'Additional info for transaction 4');
INSERT INTO "Transaction_Details" VALUES(5,5,'Additional info for transaction 5');
CREATE TABLE Transaction_Types (
    TypeID INTEGER PRIMARY KEY AUTOINCREMENT,
    TypeName TEXT
);
INSERT INTO "Transaction_Types" VALUES(1,'Payment');
INSERT INTO "Transaction_Types" VALUES(2,'Transfer');
INSERT INTO "Transaction_Types" VALUES(3,'Withdrawal');
INSERT INTO "Transaction_Types" VALUES(4,'Deposit');
INSERT INTO "Transaction_Types" VALUES(5,'Remittance Transfer');
CREATE TABLE "Transactions" (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Amount REAL ,
    DateAndTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    TypeID INTEGER,
    FromAccountId INTEGER,
    ToAccountId INTEGER,
    FOREIGN KEY (TypeID) REFERENCES Transaction_Types(TypeID),
    FOREIGN KEY (FromAccountId) REFERENCES Accounts(AccountID),
    FOREIGN KEY (ToAccountId) REFERENCES Accounts(AccountID)
);
INSERT INTO "Transactions" VALUES(1,500.0,'2024-03-13 14:46:32',1,1,2);
INSERT INTO "Transactions" VALUES(2,600.0,'2024-03-13 14:46:32',2,3,4);
INSERT INTO "Transactions" VALUES(3,-400.0,'2024-03-13 14:46:32',3,1,1);
INSERT INTO "Transactions" VALUES(4,700.0,'2024-03-13 14:46:32',4,3,5);
INSERT INTO "Transactions" VALUES(5,800.0,'2024-03-13 14:46:32',1,2,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Transaction_Types',5);
INSERT INTO "sqlite_sequence" VALUES('Banks',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transaction_Details',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
COMMIT;
