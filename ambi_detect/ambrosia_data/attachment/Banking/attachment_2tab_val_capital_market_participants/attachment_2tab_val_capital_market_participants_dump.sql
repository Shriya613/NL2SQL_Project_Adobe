BEGIN TRANSACTION;
CREATE TABLE Accounts (
    account_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    balance REAL,
    interest_rate DECIMAL(5,2) ,
    FOREIGN KEY (customer_id) REFERENCES Customers(id)
);
INSERT INTO "Accounts" VALUES(1,1,1000.0,2.5);
INSERT INTO "Accounts" VALUES(2,2,2000.0,3);
INSERT INTO "Accounts" VALUES(3,3,3000.0,2);
INSERT INTO "Accounts" VALUES(4,4,4000.0,3.5);
CREATE TABLE Customers (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    phone_number TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','1990-01-01','123-456-7890','john@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1992-02-02','987-654-3210','jane@example.com');
INSERT INTO "Customers" VALUES(3,'Alice','Johnson','1988-03-03','555-555-5555','alice@example.com');
INSERT INTO "Customers" VALUES(4,'Bob','Brown','1991-04-04','111-222-3333','bob@example.com');
CREATE TABLE Investments (
    investment_id INTEGER PRIMARY KEY,
    stockbroker_id INTEGER,
    mutual_fund_id INTEGER,
    customer_id INTEGER,
    quantity INTEGER,
    purchase_price REAL,
    FOREIGN KEY (stockbroker_id) REFERENCES Stockbrokers(id),
    FOREIGN KEY (mutual_fund_id) REFERENCES Mutual_funds(id),
    FOREIGN KEY (customer_id) REFERENCES Customers(id)
);
INSERT INTO "Investments" VALUES(1,1,1,1,10,100.0);
INSERT INTO "Investments" VALUES(2,1,2,2,20,200.0);
INSERT INTO "Investments" VALUES(3,2,1,3,5,150.0);
INSERT INTO "Investments" VALUES(4,2,2,4,10,250.0);
INSERT INTO "Investments" VALUES(5,3,3,1,20,300.0);
INSERT INTO "Investments" VALUES(6,3,4,2,15,350.0);
CREATE TABLE Mutual_funds (
    id INTEGER PRIMARY KEY,
    fund_name TEXT,
    total_assets REAL,
    expense_ratio DECIMAL(8,2) ,
    fees VARCHAR(5) 
);
INSERT INTO "Mutual_funds" VALUES(1,'ABC Fund',1000000.0,0.01,'1%');
INSERT INTO "Mutual_funds" VALUES(2,'XYZ Fund',500000.0,0.02,'1.25%');
INSERT INTO "Mutual_funds" VALUES(3,'PQR Fund',200000.0,0.03,'2%');
INSERT INTO "Mutual_funds" VALUES(4,'MNO Fund',800000.0,0.005,'0.5%');
CREATE TABLE Stockbrokers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    contact_number TEXT,
    email TEXT,
    fees VARCHAR(5) 
);
INSERT INTO "Stockbrokers" VALUES(1,'John Doe','123-456-7890','john@example.com','1%');
INSERT INTO "Stockbrokers" VALUES(2,'Jane Smith','987-654-3210','jane@example.com','2%');
INSERT INTO "Stockbrokers" VALUES(3,'Alice Johnson','555-555-5555','alice@example.com','1.5%');
INSERT INTO "Stockbrokers" VALUES(4,'Bob Brown','111-222-3333','bob@example.com','0.75%');
CREATE TABLE Transactions (
    transaction_id INTEGER PRIMARY KEY,
    account_id INTEGER,
    amount REAL,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES Accounts(account_id)
);
INSERT INTO "Transactions" VALUES(1,1,500.0,'2022-01-01 10:00:00');
INSERT INTO "Transactions" VALUES(2,2,1000.0,'2022-02-01 11:00:00');
INSERT INTO "Transactions" VALUES(3,3,1500.0,'2022-03-01 12:00:00');
INSERT INTO "Transactions" VALUES(4,4,2000.0,'2022-04-01 13:00:00');
COMMIT;
