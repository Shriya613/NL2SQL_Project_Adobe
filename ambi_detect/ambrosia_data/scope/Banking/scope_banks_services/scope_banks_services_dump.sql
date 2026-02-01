BEGIN TRANSACTION;
CREATE TABLE Accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accountNumber TEXT,
    balance REAL  DEFAULT 0,
    customerId INTEGER,
    FOREIGN KEY (customerId) REFERENCES Customers(id)
);
INSERT INTO "Accounts" VALUES(1,'123456789',10000.0,1);
INSERT INTO "Accounts" VALUES(2,'987654321',5000.0,2);
INSERT INTO "Accounts" VALUES(3,'543210987',7000.0,3);
INSERT INTO "Accounts" VALUES(4,'654321098',20000.0,4);
INSERT INTO "Accounts" VALUES(5,'876543210',8000.0,5);
CREATE TABLE Banks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    headquarter TEXT
);
INSERT INTO "Banks" VALUES(1,'Bank A','New York');
INSERT INTO "Banks" VALUES(2,'Bank B','London');
INSERT INTO "Banks" VALUES(3,'Bank C','Tokyo');
INSERT INTO "Banks" VALUES(4,'Bank D','Paris');
INSERT INTO "Banks" VALUES(5,'Bank E','Dubai');
CREATE TABLE Banks_Services (
    bankId INTEGER,
    serviceId INTEGER,
    FOREIGN KEY (bankId) REFERENCES Banks(id),
    FOREIGN KEY (serviceId) REFERENCES Services(id),
    PRIMARY KEY (bankId, serviceId)
);
INSERT INTO "Banks_Services" VALUES(1,1);
INSERT INTO "Banks_Services" VALUES(1,2);
INSERT INTO "Banks_Services" VALUES(2,1);
INSERT INTO "Banks_Services" VALUES(2,2);
INSERT INTO "Banks_Services" VALUES(3,3);
INSERT INTO "Banks_Services" VALUES(3,4);
INSERT INTO "Banks_Services" VALUES(3,1);
INSERT INTO "Banks_Services" VALUES(4,1);
INSERT INTO "Banks_Services" VALUES(5,1);
CREATE TABLE Customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE
);
INSERT INTO "Customers" VALUES(1,'John','Smith','1987-03-25');
INSERT INTO "Customers" VALUES(2,'Jane','Doe','1992-06-10');
INSERT INTO "Customers" VALUES(3,'Peter','Black','1980-10-15');
INSERT INTO "Customers" VALUES(4,'Emily','White','1995-02-20');
INSERT INTO "Customers" VALUES(5,'David','Green','1977-08-25');
CREATE TABLE Services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    serviceName TEXT
);
INSERT INTO "Services" VALUES(1,'Savings_Account');
INSERT INTO "Services" VALUES(2,'Current_Account');
INSERT INTO "Services" VALUES(3,'Loan');
INSERT INTO "Services" VALUES(4,'Credit_Card');
INSERT INTO "Services" VALUES(5,'Mortgage');
CREATE TABLE Transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transactionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    amount REAL ,
    description TEXT,
    accountId INTEGER,
    FOREIGN KEY (accountId) REFERENCES Accounts(id)
);
INSERT INTO "Transactions" VALUES(1,'2022-01-01 10:00:00',500.0,'Payment',1);
INSERT INTO "Transactions" VALUES(2,'2022-01-02 15:30:00',-300.0,'Withdrawal',2);
INSERT INTO "Transactions" VALUES(3,'2022-01-03 09:15:00',700.0,'Deposit',3);
INSERT INTO "Transactions" VALUES(4,'2022-01-04 12:45:00',-400.0,'Transfer',4);
INSERT INTO "Transactions" VALUES(5,'2022-01-05 17:00:00',900.0,'Online Purchase',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Banks',5);
INSERT INTO "sqlite_sequence" VALUES('Services',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
COMMIT;
