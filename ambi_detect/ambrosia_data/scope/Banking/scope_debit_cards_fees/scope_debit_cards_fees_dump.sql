BEGIN TRANSACTION;
CREATE TABLE Debit_Card_Fees(
    CardID INTEGER,
    FeeType TEXT,
    AmountPaid REAL,
    FOREIGN KEY (CardID) REFERENCES Debit_Cards(CardID),
    FOREIGN KEY (FeeType) REFERENCES Fees(FeeType)
);
INSERT INTO "Debit_Card_Fees" VALUES(2,'Overdraft_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(2,'Monthly_Maintenance_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(3,'Overdraft_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(3,'Monthly_Maintenance_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(4,'Overdraft_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(4,'Monthly_Maintenance_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(5,'Overdraft_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(5,'Monthly_Maintenance_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(1,'Foreign_Exchange_Fee',NULL);
INSERT INTO "Debit_Card_Fees" VALUES(2,'Foreign_Exchange_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(3,'Foreign_Exchange_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(4,'Foreign_Exchange_Fee',0.0);
INSERT INTO "Debit_Card_Fees" VALUES(5,'Foreign_Exchange_Fee',0.0);
CREATE TABLE Debit_Cards(
    CardID INTEGER PRIMARY KEY,
    CustomerName TEXT,
    AccountBalance REAL 
);
INSERT INTO "Debit_Cards" VALUES(1,'John Doe',2000.0);
INSERT INTO "Debit_Cards" VALUES(2,'Jane Smith',3000.0);
INSERT INTO "Debit_Cards" VALUES(3,'Mike Johnson',1500.0);
INSERT INTO "Debit_Cards" VALUES(4,'Emily Davis',4000.0);
INSERT INTO "Debit_Cards" VALUES(5,'David Brown',5000.0);
CREATE TABLE Fees(
    FeeType TEXT PRIMARY KEY,
    FeeValue REAL 
);
INSERT INTO "Fees" VALUES('Overdraft_Fee',35.0);
INSERT INTO "Fees" VALUES('Monthly_Maintenance_Fee',10.0);
INSERT INTO "Fees" VALUES('Transaction_Fee',0.5);
INSERT INTO "Fees" VALUES('Foreign_Exchange_Fee',3.0);
INSERT INTO "Fees" VALUES('ATM_Withdrawal_Fee',2.0);
COMMIT;
