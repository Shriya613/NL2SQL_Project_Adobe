BEGIN TRANSACTION;
CREATE TABLE Accounts (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    account_type TEXT,
    balance REAL ,
    FOREIGN KEY (customer_id) REFERENCES Customers (customer_id) ON DELETE CASCADE
);
INSERT INTO "Accounts" VALUES(1,NULL,'Checking',1000.0);
INSERT INTO "Accounts" VALUES(2,NULL,'Savings',5000.0);
INSERT INTO "Accounts" VALUES(3,NULL,'Credit Card',2000.0);
INSERT INTO "Accounts" VALUES(4,NULL,'Loan',-1500.0);
INSERT INTO "Accounts" VALUES(5,NULL,'Mortgage',-20000.0);
CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number VARCHAR(15)
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com','+123-456-7890');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com','+098-765-4321');
INSERT INTO "Customers" VALUES(3,'Michael','Brown','michael.brown@example.com','+456-789-0123');
INSERT INTO "Customers" VALUES(4,'Alice','White','alice.white@example.com','+135-790-2468');
INSERT INTO "Customers" VALUES(5,'David','Black','david.black@example.com','+246-890-1357');
CREATE TABLE Features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_value TEXT
);
INSERT INTO "Features" VALUES(1,'Mobile Deposit');
INSERT INTO "Features" VALUES(2,'Bill Pay');
INSERT INTO "Features" VALUES(3,'Person to Person Payment');
INSERT INTO "Features" VALUES(4,'Fund Transfer between accounts');
INSERT INTO "Features" VALUES(5,'Account Alerts');
CREATE TABLE Mobile_Banking_App (
    app_id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_version TEXT,
    release_date DATE,
    platform TEXT
);
INSERT INTO "Mobile_Banking_App" VALUES(1,'v1.0','2021-01-01','Android');
INSERT INTO "Mobile_Banking_App" VALUES(2,'v1.1','2021-05-01','iOS');
INSERT INTO "Mobile_Banking_App" VALUES(3,'v1.2','2021-07-01','Android');
INSERT INTO "Mobile_Banking_App" VALUES(4,'v1.3','2021-11-01','iOS');
INSERT INTO "Mobile_Banking_App" VALUES(5,'v1.4','2022-01-01','Android');
CREATE TABLE "Mobile_Banking_App_Features" (
    mobile_app_feature_id INTEGER,
    mobile_banking_app_id INTEGER,
    feature_id INTEGER,
    FOREIGN KEY (feature_id) REFERENCES Features (feature_id) ON DELETE CASCADE,
    FOREIGN KEY (mobile_banking_app_id) REFERENCES Mobile_Banking_App (app_id) ON DELETE CASCADE,
    PRIMARY KEY (mobile_app_feature_id, mobile_banking_app_id, feature_id)
);
INSERT INTO "Mobile_Banking_App_Features" VALUES(NULL,1,1);
INSERT INTO "Mobile_Banking_App_Features" VALUES(NULL,2,1);
INSERT INTO "Mobile_Banking_App_Features" VALUES(NULL,3,1);
INSERT INTO "Mobile_Banking_App_Features" VALUES(NULL,4,1);
INSERT INTO "Mobile_Banking_App_Features" VALUES(NULL,5,1);
INSERT INTO "Mobile_Banking_App_Features" VALUES(NULL,5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Accounts',5);
INSERT INTO "sqlite_sequence" VALUES('Mobile_Banking_App',5);
INSERT INTO "sqlite_sequence" VALUES('Features',5);
COMMIT;
