BEGIN TRANSACTION;
CREATE TABLE Metrics(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT,
    value REAL);
INSERT INTO "Metrics" VALUES(1,'Conversion_Rate',0.85);
INSERT INTO "Metrics" VALUES(2,'Revenue',50000.0);
INSERT INTO "Metrics" VALUES(3,'Traffic',30000.0);
INSERT INTO "Metrics" VALUES(4,'Average Order Value',75.0);
INSERT INTO "Metrics" VALUES(5,'Bounce Rate',0.15);
CREATE TABLE Product_Managers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Product_Managers" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Product_Managers" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Product_Managers" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Product_Managers" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Product_Managers" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Product_Managers_Metrics(
    product_manager_id INTEGER,
    metrics_id INTEGER,
    FOREIGN KEY(product_manager_id) REFERENCES Product_Managers(id),
    FOREIGN KEY(metrics_id) REFERENCES Metrics(id));
INSERT INTO "Product_Managers_Metrics" VALUES(1,1);
INSERT INTO "Product_Managers_Metrics" VALUES(1,2);
INSERT INTO "Product_Managers_Metrics" VALUES(2,3);
INSERT INTO "Product_Managers_Metrics" VALUES(2,4);
INSERT INTO "Product_Managers_Metrics" VALUES(3,5);
INSERT INTO "Product_Managers_Metrics" VALUES(4,1);
INSERT INTO "Product_Managers_Metrics" VALUES(5,2);
INSERT INTO "Product_Managers_Metrics" VALUES(2,1);
INSERT INTO "Product_Managers_Metrics" VALUES(3,1);
INSERT INTO "Product_Managers_Metrics" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Product_Managers',5);
INSERT INTO "sqlite_sequence" VALUES('Metrics',5);
COMMIT;
