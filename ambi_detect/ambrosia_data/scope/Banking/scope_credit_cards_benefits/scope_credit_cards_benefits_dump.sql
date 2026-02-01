BEGIN TRANSACTION;
CREATE TABLE Accounts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customerId INTEGER,
    balance REAL DEFAULT 0,
    FOREIGN KEY (customerId) REFERENCES Customers(id));
INSERT INTO "Accounts" VALUES(1,1,0.0);
INSERT INTO "Accounts" VALUES(2,2,0.0);
INSERT INTO "Accounts" VALUES(3,3,0.0);
INSERT INTO "Accounts" VALUES(4,4,0.0);
INSERT INTO "Accounts" VALUES(5,5,0.0);
CREATE TABLE Benefits(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benefitName TEXT);
INSERT INTO "Benefits" VALUES(1,'Rewards_Program');
INSERT INTO "Benefits" VALUES(2,'Travel Insurance');
INSERT INTO "Benefits" VALUES(3,'Extended Warranty');
INSERT INTO "Benefits" VALUES(4,'Purchase Protection');
INSERT INTO "Benefits" VALUES(5,'Price Protection');
CREATE TABLE Credit_Card_Benefits(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    creditCardId INTEGER,
    benefitId INTEGER,
    FOREIGN KEY (creditCardId) REFERENCES Credit_Cards(id),
    FOREIGN KEY (benefitId) REFERENCES Benefits(id));
INSERT INTO "Credit_Card_Benefits" VALUES(1,1,1);
INSERT INTO "Credit_Card_Benefits" VALUES(2,2,1);
INSERT INTO "Credit_Card_Benefits" VALUES(3,3,1);
INSERT INTO "Credit_Card_Benefits" VALUES(4,4,1);
INSERT INTO "Credit_Card_Benefits" VALUES(5,5,1);
INSERT INTO "Credit_Card_Benefits" VALUES(6,5,4);
CREATE TABLE Credit_Cards(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accountId INTEGER,
    cardNumber TEXT,
    expiryDate DATE,
    securityCode INT,
    creditLimit REAL DEFAULT 5000,
    FOREIGN KEY (accountId) REFERENCES Accounts(id));
INSERT INTO "Credit_Cards" VALUES(1,1,'1234567890123456','2023-12-31',123,5000.0);
INSERT INTO "Credit_Cards" VALUES(2,2,'0987654321098765','2024-12-31',456,5000.0);
INSERT INTO "Credit_Cards" VALUES(3,3,'1122334455667788','2025-12-31',789,5000.0);
INSERT INTO "Credit_Cards" VALUES(4,4,'6655443322110099','2026-12-31',0,5000.0);
INSERT INTO "Credit_Cards" VALUES(5,5,'9988776655443322','2027-12-31',111,5000.0);
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Customers" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Customers" VALUES(3,'Michael Johnson','michaeljohnson@example.com');
INSERT INTO "Customers" VALUES(4,'Emily Davis','emilydavis@example.com');
INSERT INTO "Customers" VALUES(5,'David Brown','davidbrown@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Benefits',5);
INSERT INTO "sqlite_sequence" VALUES('Credit_Cards',5);
INSERT INTO "sqlite_sequence" VALUES('Credit_Card_Benefits',6);
COMMIT;
