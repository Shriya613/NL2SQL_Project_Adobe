BEGIN TRANSACTION;
CREATE TABLE Business_Loan_Applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_name TEXT,
    application_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    annual_revenue REAL ,
    num_employees INTEGER ,
    loan_id INTEGER,
    FOREIGN KEY(loan_id) REFERENCES Loans(id)
);
INSERT INTO "Business_Loan_Applications" VALUES(1,'Start-up Loan','2024-02-19 14:54:05',200000.0,10,1);
INSERT INTO "Business_Loan_Applications" VALUES(2,'Small Business Expansion Loan','2024-02-19 14:54:05',500000.0,50,2);
INSERT INTO "Business_Loan_Applications" VALUES(3,'Equipment Financing Loan','2024-02-19 14:54:05',150000.0,25,3);
INSERT INTO "Business_Loan_Applications" VALUES(4,'Line of Credit','2024-02-19 14:54:05',120000.0,30,4);
INSERT INTO "Business_Loan_Applications" VALUES(5,'Commercial Real Estate Loan','2024-02-19 14:54:05',1000000.0,100,5);
CREATE TABLE "Loans" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    amount REAL ,
    term INTEGER ,
    eligibility_criteria_id INTEGER,
    FOREIGN KEY(eligibility_criteria_id) REFERENCES "Requirements"(id));
INSERT INTO "Loans" VALUES(1,'Start-up Loan',50000.0,12,1);
INSERT INTO "Loans" VALUES(2,'Small Business Expansion Loan',100000.0,24,2);
INSERT INTO "Loans" VALUES(3,'Equipment Financing Loan',75000.0,12,3);
INSERT INTO "Loans" VALUES(4,'Line of Credit',50000.0,12,4);
INSERT INTO "Loans" VALUES(5,'Commercial Real Estate Loan',200000.0,36,5);
CREATE TABLE "Requirements" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criteria TEXT,
    score REAL 
);
INSERT INTO "Requirements" VALUES(1,'Credit Score above 700',1.0);
INSERT INTO "Requirements" VALUES(2,'Business Revenue over $1 million',0.8);
INSERT INTO "Requirements" VALUES(3,'2 years of business operation',0.6);
INSERT INTO "Requirements" VALUES(4,'No bankruptcy history',1.0);
INSERT INTO "Requirements" VALUES(5,'Collateral available',0.9);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Requirements',5);
INSERT INTO "sqlite_sequence" VALUES('Business_Loan_Applications',5);
INSERT INTO "sqlite_sequence" VALUES('Loans',5);
COMMIT;
