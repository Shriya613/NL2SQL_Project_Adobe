BEGIN TRANSACTION;
CREATE TABLE Account (
    id INTEGER PRIMARY KEY,
    account_number INTEGER,
    balance INTEGER DEFAULT 0,
    commercial_bank_id INTEGER REFERENCES Commercial_Bank(id),
    FOREIGN KEY(commercial_bank_id) REFERENCES Commercial_Bank(id));
INSERT INTO "Account" VALUES(1,123456789,0,1);
INSERT INTO "Account" VALUES(2,987654321,0,2);
CREATE TABLE Central_Bank (
    id INTEGER PRIMARY KEY,
    central_bank_name TEXT,
    country_id INTEGER REFERENCES Country(id));
INSERT INTO "Central_Bank" VALUES(1,'Federal Reserve',1);
INSERT INTO "Central_Bank" VALUES(2,'Bank of England',2);
CREATE TABLE Commercial_Bank (
    id INTEGER PRIMARY KEY,
    commercial_bank_name TEXT,
    country_id INTEGER REFERENCES Country(id));
INSERT INTO "Commercial_Bank" VALUES(1,'Citibank',1);
INSERT INTO "Commercial_Bank" VALUES(2,'Barclays',2);
CREATE TABLE Country (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE);
INSERT INTO "Country" VALUES(1,'USA');
INSERT INTO "Country" VALUES(2,'UK');
COMMIT;
