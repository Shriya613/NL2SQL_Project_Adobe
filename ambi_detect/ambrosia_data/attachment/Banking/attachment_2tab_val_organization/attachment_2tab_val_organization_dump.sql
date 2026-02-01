BEGIN TRANSACTION;
CREATE TABLE Account (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    account_type TEXT, -- e.g., 'Checking', 'Savings'
    balance REAL,
    FOREIGN KEY (customer_id) REFERENCES Customer(id));
INSERT INTO "Account" VALUES(1,1,'Checking',500.0);
INSERT INTO "Account" VALUES(2,2,'Savings',1000.0);
CREATE TABLE Commodity_Trading_Firm (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Year_of_establishment INT ,
    Number_of_employees INT,
    Annual_revenue REAL);
INSERT INTO "Commodity_Trading_Firm" VALUES(1,'CT1',2000,300,7500000.0);
INSERT INTO "Commodity_Trading_Firm" VALUES(2,'CT2',2010,200,6500000.0);
CREATE TABLE Currency_Exchange_Service (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Year_of_establishment INT ,
    Number_of_branches INT,
    Total_assets REAL);
INSERT INTO "Currency_Exchange_Service" VALUES(1,'CX1',2000,100,1000000.0);
INSERT INTO "Currency_Exchange_Service" VALUES(2,'CX2',2010,80,900000.0);
CREATE TABLE Customer (
    id INTEGER PRIMARY KEY,
    First_Name TEXT,
    Last_Name TEXT,
    Date_of_birth DATE,
    Email TEXT UNIQUE);
INSERT INTO "Customer" VALUES(1,'John','Doe','1980-01-01','john.doe@example.com');
INSERT INTO "Customer" VALUES(2,'Jane','Smith','1985-02-15','jane.smith@example.com');
CREATE TABLE Investment (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    investment_type TEXT, -- e.g., 'Stocks', 'Bonds'
    amount REAL,
    date_purchased DATE,
    current_value REAL,
    FOREIGN KEY (customer_id) REFERENCES Customer(id));
INSERT INTO "Investment" VALUES(1,1,'Stocks',20000.0,'2020-03-01',21000.0);
INSERT INTO "Investment" VALUES(2,2,'Bonds',5000.0,'2019-04-01',5500.0);
CREATE TABLE Loan (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    loan_amount REAL,
    interest_rate REAL,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (customer_id) REFERENCES Customer(id));
INSERT INTO "Loan" VALUES(1,1,150000.0,5.0,'2021-01-01','2026-01-01');
INSERT INTO "Loan" VALUES(2,2,200000.0,6.0,'2021-02-01','2027-02-01');
COMMIT;
