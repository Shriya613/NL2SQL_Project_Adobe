BEGIN TRANSACTION;
CREATE TABLE accounts(
    account_number INTEGER PRIMARY KEY,
    balance REAL DEFAULT 0,
    interest_rate REAL ,
    customer_id INTEGER,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);
INSERT INTO "accounts" VALUES(1,1000.0,2.0,1);
INSERT INTO "accounts" VALUES(2,3000.0,2.5,2);
CREATE TABLE "annuities"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_number INTEGER,
    payout REAL,
    premium_amount REAL,
    start_date DATE,
    end_date DATE,
    customer_id INTEGER,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);
INSERT INTO "annuities" VALUES(1,345,100000.0,10000.0,'2023-01-01','2033-01-01',1);
INSERT INTO "annuities" VALUES(2,456,150000.0,15000.0,'2024-01-01','2034-01-01',2);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    email TEXT UNIQUE
);
INSERT INTO "customers" VALUES(1,'John','Doe','2000-01-01','john@example.com');
INSERT INTO "customers" VALUES(2,'Jane','Smith','1985-06-30','jane@example.com');
CREATE TABLE "life_insurances"(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_number INTEGER,
    payout REAL,
    coverage_amount REAL,
    term_years INTEGER,
    customer_id INTEGER,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);
INSERT INTO "life_insurances" VALUES(1,123,100000.0,500000.0,20,1);
INSERT INTO "life_insurances" VALUES(2,234,150000.0,600000.0,20,2);
CREATE TABLE loans(
    loan_id INTEGER PRIMARY KEY,
    amount REAL,
    start_date DATE,
    end_date DATE,
    monthly_payment REAL,
    account_number INTEGER,
    FOREIGN KEY(account_number) REFERENCES accounts(account_number)
);
INSERT INTO "loans" VALUES(1,50000.0,'2022-01-01','2027-01-01',1000.0,1);
INSERT INTO "loans" VALUES(2,75000.0,'2021-01-01','2026-01-01',1200.0,2);
CREATE TABLE transactions(
    transaction_id INTEGER PRIMARY KEY,
    type TEXT ,
    amount REAL,
    account_number INTEGER,
    FOREIGN KEY(account_number) REFERENCES accounts(account_number)
);
INSERT INTO "transactions" VALUES(1,'Deposit',1000.0,1);
INSERT INTO "transactions" VALUES(2,'Withdrawal',500.0,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('life_insurances',2);
INSERT INTO "sqlite_sequence" VALUES('annuities',2);
COMMIT;
