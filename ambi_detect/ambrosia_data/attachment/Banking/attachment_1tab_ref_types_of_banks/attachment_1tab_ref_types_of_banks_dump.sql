BEGIN TRANSACTION;
CREATE TABLE Accounts(
    id INTEGER PRIMARY KEY,
    balance DOUBLE,
    customer_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES Customers(id));
INSERT INTO "Accounts" VALUES(1,1000.0,1);
INSERT INTO "Accounts" VALUES(2,2000.0,2);
CREATE TABLE Banks(
    id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    type_id INTEGER,
    country_id INTEGER,
    FOREIGN KEY (type_id) REFERENCES Types_of_banks(id),
    FOREIGN KEY (country_id) REFERENCES Countries(id));
INSERT INTO "Banks" VALUES(1,'Federal Reserve',1,1);
INSERT INTO "Banks" VALUES(2,'Bank of England',1,2);
INSERT INTO "Banks" VALUES(3,'Citibank',2,1);
INSERT INTO "Banks" VALUES(4,'Barclays',2,2);
CREATE TABLE Countries(
    id INTEGER PRIMARY KEY,
    country_name VARCHAR(255) UNIQUE);
INSERT INTO "Countries" VALUES(1,'USA');
INSERT INTO "Countries" VALUES(2,'UK');
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birthdate DATE,
    account_number INTEGER,
    bank_id INTEGER,
    FOREIGN KEY (account_number) REFERENCES Accounts(id),
    FOREIGN KEY (bank_id) REFERENCES Banks(id));
INSERT INTO "Customers" VALUES(1,'John','Doe','1980-01-01',1,1);
INSERT INTO "Customers" VALUES(2,'Jane','Smith','1985-05-15',2,2);
CREATE TABLE Types_of_banks(
    id INTEGER PRIMARY KEY,
    bank_type VARCHAR(255));
INSERT INTO "Types_of_banks" VALUES(1,'Central Bank');
INSERT INTO "Types_of_banks" VALUES(2,'Commercial Bank');
COMMIT;
