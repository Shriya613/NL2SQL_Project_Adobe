BEGIN TRANSACTION;
CREATE TABLE accounts (
    account_number INTEGER PRIMARY KEY AUTOINCREMENT,
    account_type TEXT ,
    balance REAL,
    customer_id INTEGER,
    FOREIGN KEY(customer_id) REFERENCES customers(id)
);
INSERT INTO "accounts" VALUES(1,'Checking',1000.0,1);
INSERT INTO "accounts" VALUES(2,'Savings',5000.0,2);
INSERT INTO "accounts" VALUES(3,'Money Market',20000.0,3);
INSERT INTO "accounts" VALUES(4,'CD',15000.0,4);
INSERT INTO "accounts" VALUES(5,'IRA',40000.0,5);
CREATE TABLE branches (
    branch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_manager TEXT,
    street_name TEXT,
    neighborhood TEXT,
    number_of_branches INTEGER
);
INSERT INTO "branches" VALUES(1,'John Smith','1st Ave','Uptown',1);
INSERT INTO "branches" VALUES(2,'Jane Doe','Broadway','Downtown',2);
INSERT INTO "branches" VALUES(3,'Alice Brown','Madison Ave','Midtown',3);
INSERT INTO "branches" VALUES(4,'Charlie White','Central Park West','Upper West Side',4);
INSERT INTO "branches" VALUES(5,'David Black','Washington Blvd','Lower Manhattan',5);
INSERT INTO "branches" VALUES(6,NULL,NULL,NULL,6);
CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone TEXT,
    address_id INTEGER
);
INSERT INTO "customers" VALUES(1,'John','Doe','john.doe@example.com','123-456-7890',1);
INSERT INTO "customers" VALUES(2,'Jane','Smith','jane.smith@example.com','098-765-4321',2);
INSERT INTO "customers" VALUES(3,'Alice','Brown','alice.brown@example.com','555-123-4567',3);
INSERT INTO "customers" VALUES(4,'Charlie','White','charlie.white@example.com','555-987-6543',4);
INSERT INTO "customers" VALUES(5,'David','Black','david.black@example.com','555-555-5555',5);
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    position TEXT,
    hire_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    branch_id INTEGER,
    FOREIGN KEY(branch_id) REFERENCES branches(branch_id)
);
INSERT INTO "employees" VALUES(1,'John','Doe','Branch Manager','2021-01-01 00:00:00',1);
INSERT INTO "employees" VALUES(2,'Jane','Doe','Teller','2021-02-01 00:00:00',1);
INSERT INTO "employees" VALUES(3,'Alice','Doe','Head Teller','2021-03-01 00:00:00',2);
INSERT INTO "employees" VALUES(4,'Charlie','Doe','Customer Service Representative','2021-04-01 00:00:00',3);
INSERT INTO "employees" VALUES(5,'David','Doe','Teller','2021-05-01 00:00:00',4);
CREATE TABLE loans (
    loan_number INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_amount REAL,
    interest_rate REAL,
    term INTEGER,
    due_date DATETIME,
    customer_id INTEGER,
    FOREIGN KEY(customer_id) REFERENCES customers(id)
);
INSERT INTO "loans" VALUES(1,10000.0,5.0,36,'2023-01-01 00:00:00',1);
INSERT INTO "loans" VALUES(2,15000.0,6.0,48,'2024-01-01 00:00:00',2);
INSERT INTO "loans" VALUES(3,20000.0,4.5,60,'2026-01-01 00:00:00',3);
INSERT INTO "loans" VALUES(4,25000.0,5.5,60,'2027-01-01 00:00:00',4);
INSERT INTO "loans" VALUES(5,30000.0,6.0,72,'2029-01-01 00:00:00',5);
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    amount REAL,
    description TEXT,
    account_id INTEGER,
    FOREIGN KEY(account_id) REFERENCES accounts(account_number)
);
INSERT INTO "transactions" VALUES(1,'2021-01-01 00:00:00',-500.0,'Initial Deposit',1);
INSERT INTO "transactions" VALUES(2,'2021-01-02 00:00:00',200.0,'Paycheck Deposit',1);
INSERT INTO "transactions" VALUES(3,'2021-01-03 00:00:00',-150.0,'Rent Payment',1);
INSERT INTO "transactions" VALUES(4,'2021-01-04 00:00:00',300.0,'Online Shopping',1);
INSERT INTO "transactions" VALUES(5,'2021-01-05 00:00:00',-50.0,'Groceries',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('customers',5);
INSERT INTO "sqlite_sequence" VALUES('accounts',5);
INSERT INTO "sqlite_sequence" VALUES('transactions',5);
INSERT INTO "sqlite_sequence" VALUES('loans',5);
INSERT INTO "sqlite_sequence" VALUES('branches',6);
INSERT INTO "sqlite_sequence" VALUES('employees',5);
COMMIT;
