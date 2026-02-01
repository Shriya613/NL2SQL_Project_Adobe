BEGIN TRANSACTION;
CREATE TABLE Banks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    types_of_bank TEXT,
    country TEXT);
INSERT INTO "Banks" VALUES(1,'Federal Reserve','Central Bank','USA');
INSERT INTO "Banks" VALUES(2,'Bank of England','Central Bank','UK');
INSERT INTO "Banks" VALUES(3,'JPMorgan Chase & Co.','Commercial Bank','USA');
INSERT INTO "Banks" VALUES(4,'Barclays','Commercial Bank','UK');
CREATE TABLE Branches(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_id INTEGER REFERENCES Banks(id) ON DELETE CASCADE,
    address TEXT,
    city TEXT,
    state TEXT);
INSERT INTO "Branches" VALUES(1,1,'23 Wall Street','New York City','NY');
INSERT INTO "Branches" VALUES(2,1,'700 Main St','Atlanta','GA');
INSERT INTO "Branches" VALUES(3,2,'26 The Strand','London','England');
INSERT INTO "Branches" VALUES(4,2,'8 New Broad Street','Manchester','England');
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE,
    phone_number TEXT,
    email TEXT UNIQUE);
INSERT INTO "Customers" VALUES(1,'John','Doe','1990-01-01','+1234567890','john@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1995-05-15','+0987654321','jane@example.com');
INSERT INTO "Customers" VALUES(3,'Alice','Wonder','1980-06-30','+1112223333','alice@example.com');
INSERT INTO "Customers" VALUES(4,'Charlie','Brown','1992-07-25','+4445556666','charlie@example.com');
CREATE TABLE Transactions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id INTEGER REFERENCES Branches(id) ON DELETE CASCADE,
    customer_id INTEGER REFERENCES Customers(id) ON DELETE CASCADE,
    transaction_type TEXT ,
    amount REAL,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
INSERT INTO "Transactions" VALUES(1,1,1,'Deposit',500.0,'2024-03-08 00:22:33');
INSERT INTO "Transactions" VALUES(2,2,2,'Withdrawal',-200.0,'2024-03-08 00:22:33');
INSERT INTO "Transactions" VALUES(3,1,3,'Transfer',300.0,'2024-03-08 00:22:33');
INSERT INTO "Transactions" VALUES(4,2,4,'Loan',1000.0,'2024-03-08 00:22:33');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Banks',4);
INSERT INTO "sqlite_sequence" VALUES('Branches',4);
INSERT INTO "sqlite_sequence" VALUES('Customers',4);
INSERT INTO "sqlite_sequence" VALUES('Transactions',4);
COMMIT;
