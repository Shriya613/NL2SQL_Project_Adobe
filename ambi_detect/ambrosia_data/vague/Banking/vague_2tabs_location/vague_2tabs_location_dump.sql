BEGIN TRANSACTION;
CREATE TABLE Accounts (
    account_number INTEGER PRIMARY KEY AUTOINCREMENT,
    balance REAL ,
    owner_id INTEGER,
    FOREIGN KEY(owner_id) REFERENCES Banks(id)
);
INSERT INTO "Accounts" VALUES(1,1000.0,1);
INSERT INTO "Accounts" VALUES(2,800.0,2);
INSERT INTO "Accounts" VALUES(3,1200.0,3);
INSERT INTO "Accounts" VALUES(4,900.0,4);
INSERT INTO "Accounts" VALUES(5,700.0,5);
CREATE TABLE Banks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name TEXT,
    city_id INTEGER,
    country_id INTEGER,
    FOREIGN KEY(city_id) REFERENCES City(id),
    FOREIGN KEY(country_id) REFERENCES Country(id)
);
INSERT INTO "Banks" VALUES(1,'Bank A',1,1);
INSERT INTO "Banks" VALUES(2,'Bank B',2,2);
INSERT INTO "Banks" VALUES(3,'Bank C',3,3);
INSERT INTO "Banks" VALUES(4,'Bank D',4,4);
INSERT INTO "Banks" VALUES(5,'Bank E',5,5);
CREATE TABLE City (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    country_id INTEGER
);
INSERT INTO "City" VALUES(1,'New York',1);
INSERT INTO "City" VALUES(2,'London',2);
INSERT INTO "City" VALUES(3,'Tokyo',3);
INSERT INTO "City" VALUES(4,'Paris',4);
INSERT INTO "City" VALUES(5,'Mumbai',5);
CREATE TABLE Country (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Country" VALUES(1,'USA');
INSERT INTO "Country" VALUES(2,'UK');
INSERT INTO "Country" VALUES(3,'Japan');
INSERT INTO "Country" VALUES(4,'France');
INSERT INTO "Country" VALUES(5,'India');
CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE,
    address TEXT,
    phone TEXT,
    email TEXT,
    FOREIGN KEY(customer_id) REFERENCES Accounts(owner_id)
);
INSERT INTO "Customers" VALUES(1,'John','Doe','1980-01-01','123 Main St','555-1234','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1985-06-01','456 Elm St','555-5678','jane.smith@example.com');
INSERT INTO "Customers" VALUES(3,'Alice','Williams','1990-12-01','789 Oak St','555-9012','alice.williams@example.com');
INSERT INTO "Customers" VALUES(4,'Bob','Johnson','1988-03-01','135 Pine St','555-1314','bob.johnson@example.com');
INSERT INTO "Customers" VALUES(5,'Carol','Brown','1992-09-01','246 Maple St','555-1415','carol.brown@example.com');
CREATE TABLE Transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    amount REAL ,
    debit_account_id INTEGER,
    credit_account_id INTEGER,
    FOREIGN KEY(debit_account_id) REFERENCES Accounts(account_number),
    FOREIGN KEY(credit_account_id) REFERENCES Accounts(account_number)
);
INSERT INTO "Transactions" VALUES(1,'2024-02-19 14:18:41',100.0,1,2);
INSERT INTO "Transactions" VALUES(2,'2024-02-19 14:18:41',200.0,3,4);
INSERT INTO "Transactions" VALUES(3,'2024-02-19 14:18:41',300.0,5,1);
INSERT INTO "Transactions" VALUES(4,'2024-02-19 14:18:41',400.0,2,3);
INSERT INTO "Transactions" VALUES(5,'2024-02-19 14:18:41',500.0,4,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('City',5);
INSERT INTO "sqlite_sequence" VALUES('Country',5);
INSERT INTO "sqlite_sequence" VALUES('Banks',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
COMMIT;
