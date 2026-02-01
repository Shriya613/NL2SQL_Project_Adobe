BEGIN TRANSACTION;
CREATE TABLE Accounts (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id INTEGER,
    account_type TEXT, -- could contain values like Savings, Checking, etc.
    balance REAL DEFAULT 0,
    FOREIGN KEY(client_id) REFERENCES Clients(client_id));
INSERT INTO "Accounts" VALUES(1,1,'Checking',1000.0);
INSERT INTO "Accounts" VALUES(2,2,'Savings',500.0);
CREATE TABLE Clients (
    client_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    email TEXT,
    phoneNumber TEXT);
INSERT INTO "Clients" VALUES(1,'John Doe','123 Main St','johndoe@example.com','555-1234');
INSERT INTO "Clients" VALUES(2,'Jane Smith','456 Elm St','janesmith@example.com','555-5678');
CREATE TABLE Limits (
    limit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER,
    card_limit REAL , -- Assuming that credit limits cannot be negative
    FOREIGN KEY(card_id) REFERENCES _Bank_Cards(card_id));
INSERT INTO "Limits" VALUES(1,1,10000.0);
INSERT INTO "Limits" VALUES(2,2,5000.0);
INSERT INTO "Limits" VALUES(3,3,10000.0);
INSERT INTO "Limits" VALUES(4,4,5000.0);
CREATE TABLE _Bank_Cards (
    card_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER,
    card_number TEXT,
    card_expiry DATE,
    cvv INTEGER,
    _bank_card_type TEXT, -- could contain values like Credit Card, Debit Card, Prepaid Card, etc.
    FOREIGN KEY(account_id) REFERENCES Accounts(account_id));
INSERT INTO "_Bank_Cards" VALUES(1,1,'1234567890123456','2023-12-31',123,'Credit Card');
INSERT INTO "_Bank_Cards" VALUES(2,1,'0987654321098765','2024-01-15',456,'Credit Card');
INSERT INTO "_Bank_Cards" VALUES(3,2,'7890123456789012','2023-12-31',789,'Debit Card');
INSERT INTO "_Bank_Cards" VALUES(4,2,'4567890123456789','2024-01-15',12,'Debit Card');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Clients',2);
INSERT INTO "sqlite_sequence" VALUES('Accounts',2);
INSERT INTO "sqlite_sequence" VALUES('_Bank_Cards',4);
INSERT INTO "sqlite_sequence" VALUES('Limits',4);
COMMIT;
