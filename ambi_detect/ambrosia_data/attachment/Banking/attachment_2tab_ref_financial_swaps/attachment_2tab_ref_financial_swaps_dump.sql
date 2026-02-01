BEGIN TRANSACTION;
CREATE TABLE Accounts(
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name TEXT,
    balance REAL ,
    FOREIGN KEY(bank_name) REFERENCES Banks(name));
INSERT INTO "Accounts" VALUES(1,'Royal Bank',20000.75);
INSERT INTO "Accounts" VALUES(2,'Central Bank',30000.98);
INSERT INTO "Accounts" VALUES(3,'Global Bank',40000.66);
INSERT INTO "Accounts" VALUES(4,'National Bank',50000.87);
INSERT INTO "Accounts" VALUES(5,'Commercial Bank',60000.75);
CREATE TABLE Banks(
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE);
INSERT INTO "Banks" VALUES(1,'Royal Bank');
INSERT INTO "Banks" VALUES(2,'Central Bank');
INSERT INTO "Banks" VALUES(3,'Global Bank');
INSERT INTO "Banks" VALUES(4,'National Bank');
INSERT INTO "Banks" VALUES(5,'Commercial Bank');
CREATE TABLE Currency_Swaps(
    swap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name TEXT,
    amount REAL ,
    maturity_date DATE,
    foreign_currency TEXT,
    FOREIGN KEY(bank_name) REFERENCES Banks(name));
INSERT INTO "Currency_Swaps" VALUES(1,'Royal Bank',7000.0,'2023-01-01','Euro');
INSERT INTO "Currency_Swaps" VALUES(2,'Central Bank',8000.0,'2023-01-02','Yen');
INSERT INTO "Currency_Swaps" VALUES(3,'Global Bank',9000.0,'2023-01-03','Pound Sterling');
INSERT INTO "Currency_Swaps" VALUES(4,'National Bank',10000.0,'2023-01-04','Dollar');
INSERT INTO "Currency_Swaps" VALUES(5,'Commercial Bank',11000.0,'2023-01-05','Franc');
CREATE TABLE Interest_Rate_Swaps(
    swap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name TEXT,
    start_date DATE,
    end_date DATE,
    fixed_rate REAL ,
    floating_rate REAL ,
    FOREIGN KEY(bank_name) REFERENCES Banks(name));
INSERT INTO "Interest_Rate_Swaps" VALUES(1,'Royal Bank','2023-01-01','2024-01-01',2.5,2.6);
INSERT INTO "Interest_Rate_Swaps" VALUES(2,'Central Bank','2023-01-02','2024-01-02',2.7,2.8);
INSERT INTO "Interest_Rate_Swaps" VALUES(3,'Global Bank','2023-01-03','2024-01-03',2.9,3.0);
INSERT INTO "Interest_Rate_Swaps" VALUES(4,'National Bank','2023-01-04','2024-01-04',3.1,3.2);
INSERT INTO "Interest_Rate_Swaps" VALUES(5,'Commercial Bank','2023-01-05','2024-01-05',3.3,3.4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Currency_Swaps',5);
INSERT INTO "sqlite_sequence" VALUES('Interest_Rate_Swaps',5);
COMMIT;
