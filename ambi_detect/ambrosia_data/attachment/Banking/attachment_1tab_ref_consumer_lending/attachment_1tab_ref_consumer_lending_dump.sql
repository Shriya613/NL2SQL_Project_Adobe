BEGIN TRANSACTION;
CREATE TABLE customers(
    id INTEGER PRIMARY KEY,
    email TEXT,
    phoneNumber INT
, FullName TEXT);
INSERT INTO "customers" VALUES(1,'john@example.com',1234567890,'John Doe');
INSERT INTO "customers" VALUES(2,'jane@example.com',987654321,'Jane Smith');
INSERT INTO "customers" VALUES(3,'alice@example.com',1123456789,'Alice Brown');
INSERT INTO "customers" VALUES(4,'bob@example.com',987654320,'Bob White');
INSERT INTO "customers" VALUES(5,'charlie@example.com',1234567891,'Charlie Black');
CREATE TABLE loans(
    id INTEGER PRIMARY KEY,
    customerId INTEGER,
    loanType TEXT ,
    amount REAL,
    termYears INTEGER,
    interestRateId INTEGER,
    FOREIGN KEY (customerId) REFERENCES customers(id),
    FOREIGN KEY (interestRateId) REFERENCES rates(id)
);
INSERT INTO "loans" VALUES(1,1,'Credit Card',10000.0,5,1);
INSERT INTO "loans" VALUES(2,2,'Credit Card',5000.0,3,2);
INSERT INTO "loans" VALUES(3,3,'Personal Loan',20000.0,10,1);
INSERT INTO "loans" VALUES(4,4,'Personal Loan',15000.0,8,2);
INSERT INTO "loans" VALUES(5,5,'Mortgage',250000.0,25,3);
INSERT INTO "loans" VALUES(6,6,'Car Loan',20000.0,5,4);
INSERT INTO "loans" VALUES(7,7,'Student Loan',10000.0,10,5);
INSERT INTO "loans" VALUES(8,8,'Business Loan',50000.0,7,6);
INSERT INTO "loans" VALUES(9,9,'Home Equity Line of Credit',30000.0,10,7);
CREATE TABLE rates(
    id INTEGER PRIMARY KEY,
    interestRate REAL
);
INSERT INTO "rates" VALUES(1,15.0);
INSERT INTO "rates" VALUES(2,10.0);
INSERT INTO "rates" VALUES(3,20.0);
INSERT INTO "rates" VALUES(4,12.0);
INSERT INTO "rates" VALUES(5,17.0);
COMMIT;
