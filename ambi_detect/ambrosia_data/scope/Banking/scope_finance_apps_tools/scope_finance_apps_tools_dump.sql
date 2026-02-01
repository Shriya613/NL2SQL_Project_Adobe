BEGIN TRANSACTION;
CREATE TABLE Finance_Apps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Finance_Apps" VALUES(1,'BankA','Online banking service');
INSERT INTO "Finance_Apps" VALUES(2,'BankB','Mobile banking app');
INSERT INTO "Finance_Apps" VALUES(3,'MoneyBox','Personal financial management tool');
INSERT INTO "Finance_Apps" VALUES(4,'CreditCardA','Digital credit card services');
INSERT INTO "Finance_Apps" VALUES(5,'InvestmentA','Stock trading platform');
CREATE TABLE Finance_Apps_Tools (
    finance_app_id INTEGER,
    tool_id INTEGER,
    FOREIGN KEY(finance_app_id) REFERENCES Finance_Apps(id),
    FOREIGN KEY(tool_id) REFERENCES Tools(id),
    PRIMARY KEY(finance_app_id, tool_id)
);
INSERT INTO "Finance_Apps_Tools" VALUES(1,1);
INSERT INTO "Finance_Apps_Tools" VALUES(2,2);
INSERT INTO "Finance_Apps_Tools" VALUES(3,3);
INSERT INTO "Finance_Apps_Tools" VALUES(4,4);
INSERT INTO "Finance_Apps_Tools" VALUES(5,5);
INSERT INTO "Finance_Apps_Tools" VALUES(2,1);
INSERT INTO "Finance_Apps_Tools" VALUES(3,1);
INSERT INTO "Finance_Apps_Tools" VALUES(4,1);
INSERT INTO "Finance_Apps_Tools" VALUES(5,1);
CREATE TABLE Tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT,
    category TEXT
);
INSERT INTO "Tools" VALUES(1,'Budget_Tracker','Expense Tracker');
INSERT INTO "Tools" VALUES(2,'Bill_Reminder','Payment Reminders');
INSERT INTO "Tools" VALUES(3,'Interest_Calculator','Loan Calculation');
INSERT INTO "Tools" VALUES(4,'Currency_Converter','Exchange Rates');
INSERT INTO "Tools" VALUES(5,'Tax_Return','Tax Filings');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Finance_Apps',5);
INSERT INTO "sqlite_sequence" VALUES('Tools',5);
COMMIT;
