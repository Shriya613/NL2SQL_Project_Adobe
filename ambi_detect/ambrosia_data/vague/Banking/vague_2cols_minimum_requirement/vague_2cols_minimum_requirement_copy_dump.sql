BEGIN TRANSACTION;
CREATE TABLE accounts(
   account_number INTEGER PRIMARY KEY,
   customer_id INTEGER,
   account_type TEXT,
   balance REAL,
   FOREIGN KEY (customer_id) REFERENCES customers(customer_id));
INSERT INTO "accounts" VALUES(1,1,'Checking',1000.0);
INSERT INTO "accounts" VALUES(2,2,'Savings',500.0);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    address TEXT);
INSERT INTO "customers" VALUES(1,'John','Doe','1970-06-15','123 Main St');
INSERT INTO "customers" VALUES(2,'Jane','Smith','1985-09-20','456 Elm St');
CREATE TABLE loans(
    loan_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    loan_amount REAL,
    interest_rate FLOAT,
    repayment_term INTEGER,
    start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id));
INSERT INTO "loans" VALUES(1,1,10000.0,5.0,36,'2024-02-19 13:39:22');
INSERT INTO "loans" VALUES(2,2,5000.0,4.0,24,'2024-02-19 13:39:22');
CREATE TABLE "minimum_requirements"(
    requirement_id INTEGER PRIMARY KEY,
    account_number INTEGER,
    Minimum_Requirement DECIMAL(8,2),
    money_market_accounts BOOLEAN,
    FOREIGN KEY (account_number) REFERENCES accounts(account_number));
INSERT INTO "minimum_requirements" VALUES(1,1,1000,0);
INSERT INTO "minimum_requirements" VALUES(2,2,500,1);
CREATE TABLE overdraft_facilities(
    facility_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    limit_amount REAL,
    interest_rate FLOAT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id));
INSERT INTO "overdraft_facilities" VALUES(1,1,200.0,10.0);
INSERT INTO "overdraft_facilities" VALUES(2,2,500.0,15.0);
CREATE TABLE transactions(
    transaction_id INTEGER PRIMARY KEY,
    account_number INTEGER,
    amount REAL,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_number) REFERENCES accounts(account_number));
INSERT INTO "transactions" VALUES(1,1,200.0,'2024-02-19 13:39:22');
INSERT INTO "transactions" VALUES(2,2,-50.0,'2024-02-19 13:39:22');
COMMIT;
