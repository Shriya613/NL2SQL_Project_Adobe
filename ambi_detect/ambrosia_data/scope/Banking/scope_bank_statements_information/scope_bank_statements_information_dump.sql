BEGIN TRANSACTION;
CREATE TABLE Accounts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    balance REAL 
);
INSERT INTO "Accounts" VALUES(1,'John Doe',1000.0);
INSERT INTO "Accounts" VALUES(2,'Jane Smith',2000.0);
INSERT INTO "Accounts" VALUES(3,'Alice Johnson',3000.0);
INSERT INTO "Accounts" VALUES(4,'Bob Brown',4000.0);
INSERT INTO "Accounts" VALUES(5,'Charlie Davis',5000.0);
CREATE TABLE Bank_Statement_Information(
    bank_statement_info_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_statements_id INTEGER,
    information_id INTEGER,
    FOREIGN KEY (bank_statements_id) REFERENCES Bank_Statements(statement_id),
    FOREIGN KEY (information_id) REFERENCES Information(info_id)
);
INSERT INTO "Bank_Statement_Information" VALUES(1,1,1);
INSERT INTO "Bank_Statement_Information" VALUES(2,1,2);
INSERT INTO "Bank_Statement_Information" VALUES(3,2,3);
INSERT INTO "Bank_Statement_Information" VALUES(4,2,4);
INSERT INTO "Bank_Statement_Information" VALUES(5,3,5);
INSERT INTO "Bank_Statement_Information" VALUES(6,2,1);
INSERT INTO "Bank_Statement_Information" VALUES(7,3,1);
INSERT INTO "Bank_Statement_Information" VALUES(8,4,1);
INSERT INTO "Bank_Statement_Information" VALUES(9,5,1);
CREATE TABLE Bank_Statements(
    statement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER,
    transaction_id INTEGER,
    FOREIGN KEY (account_id) REFERENCES Accounts(id),
    FOREIGN KEY (transaction_id) REFERENCES Transactions(id)
);
INSERT INTO "Bank_Statements" VALUES(1,1,1);
INSERT INTO "Bank_Statements" VALUES(2,2,2);
INSERT INTO "Bank_Statements" VALUES(3,3,3);
INSERT INTO "Bank_Statements" VALUES(4,4,4);
INSERT INTO "Bank_Statements" VALUES(5,5,5);
CREATE TABLE Information(
    info_id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT,
    value REAL
);
INSERT INTO "Information" VALUES(1,'Transaction History','Transaction 1 Description');
INSERT INTO "Information" VALUES(2,'Other Value',1234.56);
INSERT INTO "Information" VALUES(3,'Transaction History','Transaction 2 Description');
INSERT INTO "Information" VALUES(4,'Other Value',5678.9);
INSERT INTO "Information" VALUES(5,'Transaction History','Transaction 3 Description');
CREATE TABLE Transactions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE,
    amount REAL,
    type TEXT 
);
INSERT INTO "Transactions" VALUES(1,'2022-01-01',100.0,'Deposit');
INSERT INTO "Transactions" VALUES(2,'2022-01-02',-50.0,'Withdrawal');
INSERT INTO "Transactions" VALUES(3,'2022-01-03',75.0,'Transfer');
INSERT INTO "Transactions" VALUES(4,'2022-01-04',-25.0,'Fee');
INSERT INTO "Transactions" VALUES(5,'2022-01-05',50.0,'Interest');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
INSERT INTO "sqlite_sequence" VALUES('Information',5);
INSERT INTO "sqlite_sequence" VALUES('Bank_Statements',5);
INSERT INTO "sqlite_sequence" VALUES('Bank_Statement_Information',9);
COMMIT;
