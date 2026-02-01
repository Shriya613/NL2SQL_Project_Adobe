BEGIN TRANSACTION;
CREATE TABLE Approaches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    approach_value TEXT
);
INSERT INTO "Approaches" VALUES(1,'Direct Mail');
INSERT INTO "Approaches" VALUES(2,'Email Campaign');
INSERT INTO "Approaches" VALUES(3,'Telephone Call');
INSERT INTO "Approaches" VALUES(4,'Online Advertising');
INSERT INTO "Approaches" VALUES(5,'Print Media');
CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john@gmail.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane@yahoo.com');
INSERT INTO "Customers" VALUES(3,'Mike','Brown','mike@hotmail.com');
INSERT INTO "Customers" VALUES(4,'Emma','White','emma@outlook.com');
INSERT INTO "Customers" VALUES(5,'Robert','Black','robert@aol.com');
CREATE TABLE Marketing_Campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_name TEXT,
    start_date DATE,
    end_date DATE
);
INSERT INTO "Marketing_Campaigns" VALUES(1,'Spring Sale','2022-05-01','2022-05-31');
INSERT INTO "Marketing_Campaigns" VALUES(2,'Summer Promotion','2022-06-01','2022-06-30');
INSERT INTO "Marketing_Campaigns" VALUES(3,'Autumn Event','2022-09-01','2022-09-30');
INSERT INTO "Marketing_Campaigns" VALUES(4,'Winter Offer','2022-11-01','2022-11-30');
INSERT INTO "Marketing_Campaigns" VALUES(5,'New Year Special','2023-01-01','2023-01-31');
CREATE TABLE Marketing_Campaigns_Approaches (
    marketing_campaign_id INTEGER,
    approach_id INTEGER,
    FOREIGN KEY (marketing_campaign_id) REFERENCES Marketing_Campaigns(id),
    FOREIGN KEY (approach_id) REFERENCES Approaches(id),
    PRIMARY KEY (marketing_campaign_id, approach_id)
);
INSERT INTO "Marketing_Campaigns_Approaches" VALUES(1,1);
INSERT INTO "Marketing_Campaigns_Approaches" VALUES(2,1);
INSERT INTO "Marketing_Campaigns_Approaches" VALUES(3,1);
INSERT INTO "Marketing_Campaigns_Approaches" VALUES(4,1);
INSERT INTO "Marketing_Campaigns_Approaches" VALUES(5,1);
INSERT INTO "Marketing_Campaigns_Approaches" VALUES(5,4);
CREATE TABLE Transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount DECIMAL(10,2) ,
    date DATE,
    customer_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
INSERT INTO "Transactions" VALUES(1,300.78,'2022-06-01',1);
INSERT INTO "Transactions" VALUES(2,500.99,'2022-06-02',2);
INSERT INTO "Transactions" VALUES(3,200.66,'2022-06-03',3);
INSERT INTO "Transactions" VALUES(4,100.45,'2022-06-04',4);
INSERT INTO "Transactions" VALUES(5,700.25,'2022-06-05',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
INSERT INTO "sqlite_sequence" VALUES('Approaches',5);
INSERT INTO "sqlite_sequence" VALUES('Marketing_Campaigns',5);
COMMIT;
