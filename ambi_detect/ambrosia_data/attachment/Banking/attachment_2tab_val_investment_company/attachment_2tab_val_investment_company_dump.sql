BEGIN TRANSACTION;
CREATE TABLE customers(
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    address TEXT,
    contact_person_name TEXT,
    phone_number CHAR(12),
    email TEXT,
    equity_firm_id INTEGER,
    venture_capital_id INTEGER,
    FOREIGN KEY (equity_firm_id) REFERENCES growth_equity(id),
    FOREIGN KEY (venture_capital_id) REFERENCES venture_capital(id));
INSERT INTO "customers" VALUES(1,'Customer 1','123 Main St','John Doe','555-1234','john@example.com',1,1);
INSERT INTO "customers" VALUES(2,'Customer 2','456 Elm St','Jane Smith','555-5678','jane@example.com',2,2);
INSERT INTO "customers" VALUES(3,'Customer 3','789 Oak St','Michael Johnson','555-9101','michael@example.com',3,3);
INSERT INTO "customers" VALUES(4,'Customer 4','246 Pine St','Emily Davis','555-1314','emily@example.com',4,4);
INSERT INTO "customers" VALUES(5,'Customer 5','678 Fir St','David Brown','555-2425','david@example.com',5,5);
CREATE TABLE employees(
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    hire_date DATE,
    position TEXT,
    assigned_customer_id INTEGER,
    FOREIGN KEY (assigned_customer_id) REFERENCES customers(customer_id));
INSERT INTO "employees" VALUES(1,'Alice','2023-01-15','Account Manager',1);
INSERT INTO "employees" VALUES(2,'Bob','2023-02-15','Business Analyst',2);
INSERT INTO "employees" VALUES(3,'Charlie','2023-03-15','Financial Advisor',3);
INSERT INTO "employees" VALUES(4,'Diana','2023-04-15','Marketing Specialist',4);
INSERT INTO "employees" VALUES(5,'Eva','2023-05-15','Data Scientist',5);
CREATE TABLE growth_equity(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firm_name TEXT,
    funding_amount REAL,
    deal_date DATE,
    notes TEXT);
INSERT INTO "growth_equity" VALUES(1,'Company A',5000000.0,'2023-06-18','Initial Investment');
INSERT INTO "growth_equity" VALUES(2,'Company B',7000000.0,'2023-06-19','Series B Funding Round');
INSERT INTO "growth_equity" VALUES(3,'Company C',2000000.0,'2023-06-20','Seed Capital');
INSERT INTO "growth_equity" VALUES(4,'Company D',4000000.0,'2023-06-21','Pre-Series A Financing');
INSERT INTO "growth_equity" VALUES(5,'Company E',9000000.0,'2023-06-22','Series C Funding');
CREATE TABLE venture_capital(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trust_name TEXT,
    funding_amount REAL,
    investment_period DATE,
    exit_strategy TEXT);
INSERT INTO "venture_capital" VALUES(1,'VC Trust 1',5000000.0,'2021-01-01','IPO Exit');
INSERT INTO "venture_capital" VALUES(2,'VC Trust 2',3000000.0,'2019-01-01','Acquisition Exit');
INSERT INTO "venture_capital" VALUES(3,'VC Trust 3',6000000.0,'2020-01-01','Merger Exit');
INSERT INTO "venture_capital" VALUES(4,'VC Trust 4',8000000.0,'2022-01-01','Secondary Share Sale');
INSERT INTO "venture_capital" VALUES(5,'VC Trust 5',10000000.0,'2024-01-01','Return of Capital');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('growth_equity',5);
INSERT INTO "sqlite_sequence" VALUES('venture_capital',5);
INSERT INTO "sqlite_sequence" VALUES('customers',5);
INSERT INTO "sqlite_sequence" VALUES('employees',5);
COMMIT;
