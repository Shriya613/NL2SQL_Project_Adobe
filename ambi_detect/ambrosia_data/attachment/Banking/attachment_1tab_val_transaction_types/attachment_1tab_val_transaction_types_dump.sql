BEGIN TRANSACTION;
CREATE TABLE Account (
    account_id INTEGER PRIMARY KEY,
    bank_account CHAR(8),
    balance REAL,
    interest_rate DECIMAL(5,2) 
);
INSERT INTO "Account" VALUES(1,'1234567',5000.0,0.05);
INSERT INTO "Account" VALUES(2,'9876543',10000.0,0.04);
INSERT INTO "Account" VALUES(3,'1111111',15000.0,0.03);
INSERT INTO "Account" VALUES(4,'2222222',20000.0,0.02);
INSERT INTO "Account" VALUES(5,'3333333',25000.0,0.01);
CREATE TABLE Customer (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT UNIQUE
);
INSERT INTO "Customer" VALUES(1,'John','Doe','john.doe@example.com','+1234567890');
INSERT INTO "Customer" VALUES(2,'Jane','Smith','jane.smith@example.com','+0987654321');
INSERT INTO "Customer" VALUES(3,'Alice','Brown','alice.brown@example.com','+1357924680');
INSERT INTO "Customer" VALUES(4,'Bob','White','bob.white@example.com','+2468013579');
INSERT INTO "Customer" VALUES(5,'Tom','Black','tom.black@example.com','+3579246801');
CREATE TABLE Employee (
    employee_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    hire_date DATE
);
INSERT INTO "Employee" VALUES(1,'Mike','Adams','mike.adams@example.com','2022-01-01');
INSERT INTO "Employee" VALUES(2,'Lisa','Wilson','lisa.wilson@example.com','2021-08-15');
INSERT INTO "Employee" VALUES(3,'David','Jackson','david.jackson@example.com','2022-05-20');
INSERT INTO "Employee" VALUES(4,'Emma','Taylor','emma.taylor@example.com','2021-12-01');
INSERT INTO "Employee" VALUES(5,'Alex','Green','alex.green@example.com','2022-03-15');
CREATE TABLE Loan (
    loan_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    loan_amount REAL,
    interest_rate DECIMAL(5,2) ,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY(customer_id) REFERENCES Customer(customer_id)
);
INSERT INTO "Loan" VALUES(1,1,'$100000',0.05,'2022-01-01','2026-01-01');
INSERT INTO "Loan" VALUES(2,2,'$150000',0.04,'2021-01-01','2025-01-01');
INSERT INTO "Loan" VALUES(3,3,'$200000',0.03,'2022-01-01','2026-01-01');
INSERT INTO "Loan" VALUES(4,4,'$250000',0.02,'2021-01-01','2025-01-01');
INSERT INTO "Loan" VALUES(5,5,'$300000',0.01,'2022-01-01','2026-01-01');
CREATE TABLE Savings_Account (
    savings_id INTEGER PRIMARY KEY,
    account_id INTEGER,
    balance REAL,
    annual_interst_rate DECIMAL(5,2) ,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(account_id) REFERENCES Account(account_id)
);
INSERT INTO "Savings_Account" VALUES(1,1,5000.0,0.05,'2022-01-01');
INSERT INTO "Savings_Account" VALUES(2,2,10000.0,0.04,'2021-01-01');
INSERT INTO "Savings_Account" VALUES(3,3,15000.0,0.03,'2022-01-01');
INSERT INTO "Savings_Account" VALUES(4,4,20000.0,0.02,'2021-01-01');
INSERT INTO "Savings_Account" VALUES(5,5,25000.0,0.01,'2022-01-01');
CREATE TABLE Transaction_Type (
    transaction_id INTEGER PRIMARY KEY,
    transaction_type VARCHAR(20),
    bank_account CHAR(8),
    amount REAL,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "Transaction_Type" VALUES(1,'Payment','1234567',1000.0,'2022-09-01');
INSERT INTO "Transaction_Type" VALUES(2,'Payment','9876543',1500.0,'2022-09-01');
INSERT INTO "Transaction_Type" VALUES(3,'Remittance Transfer','1234567',2000.0,'2022-09-01');
INSERT INTO "Transaction_Type" VALUES(4,'Remittance Transfer','9876543',3000.0,'2022-09-01');
INSERT INTO "Transaction_Type" VALUES(5,'Deposit','1234567',2500.0,'2022-09-01');
INSERT INTO "Transaction_Type" VALUES(6,'Withdrawal','9876543',500.0,'2022-09-01');
COMMIT;
