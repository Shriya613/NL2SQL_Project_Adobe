BEGIN TRANSACTION;
CREATE TABLE Accounts(
    account_id INTEGER PRIMARY KEY,
    cust_id INTEGER,
    balance REAL DEFAULT 0,
    FOREIGN KEY (cust_id) REFERENCES Customers(cust_id));
INSERT INTO "Accounts" VALUES(1,1,1000.0);
INSERT INTO "Accounts" VALUES(2,2,5000.0);
INSERT INTO "Accounts" VALUES(3,3,2000.0);
INSERT INTO "Accounts" VALUES(4,4,3000.0);
INSERT INTO "Accounts" VALUES(5,5,7000.0);
CREATE TABLE Customers(
    cust_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT,
    emp_id INTEGER,
    FOREIGN KEY (emp_id) REFERENCES Employees(emp_id));
CREATE TABLE Employees(
    emp_id INTEGER PRIMARY KEY,
    position TEXT,
    Board_Member TEXT ,
    Organization TEXT, FullName TEXT);
INSERT INTO "Employees" VALUES(1,'CEO','Executive Manager','Global Corp','John Doe');
INSERT INTO "Employees" VALUES(2,'CFO','Executive Manager','Global Corp','Jane Smith');
INSERT INTO "Employees" VALUES(3,'COO','Independent Director','Global Corp','Mike Brown');
INSERT INTO "Employees" VALUES(4,'CTO','Independent Director','Not Global Corp','Alice Green');
CREATE TABLE Transactions(
    transaction_id INTEGER PRIMARY KEY,
    account_id INTEGER,
    amount REAL,
    date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES Accounts(account_id));
INSERT INTO "Transactions" VALUES(1,1,100.0,'2021-01-01 12:00:00');
INSERT INTO "Transactions" VALUES(2,1,-50.0,'2021-01-02 12:00:00');
INSERT INTO "Transactions" VALUES(3,2,200.0,'2021-01-01 13:00:00');
INSERT INTO "Transactions" VALUES(4,2,-100.0,'2021-01-02 13:00:00');
INSERT INTO "Transactions" VALUES(5,3,200.0,'2021-01-01 14:00:00');
COMMIT;
