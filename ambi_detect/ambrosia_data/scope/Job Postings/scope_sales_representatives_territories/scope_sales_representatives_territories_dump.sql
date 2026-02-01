BEGIN TRANSACTION;
CREATE TABLE Company(
    CompanyID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Industry TEXT);
INSERT INTO "Company" VALUES(1,'ABCD Co.','Technology');
INSERT INTO "Company" VALUES(2,'EFGH Inc.','Healthcare');
INSERT INTO "Company" VALUES(3,'IJKL Ltd.','Finance');
INSERT INTO "Company" VALUES(4,'MNOP PLC.','Retail');
INSERT INTO "Company" VALUES(5,'STUV AG.','Automotive');
CREATE TABLE Products(
    ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Price DECIMAL(10,2));
INSERT INTO "Products" VALUES(1,'Product A',9.99);
INSERT INTO "Products" VALUES(2,'Product B',19.99);
INSERT INTO "Products" VALUES(3,'Product C',29.99);
INSERT INTO "Products" VALUES(4,'Product D',39.99);
INSERT INTO "Products" VALUES(5,'Product E',49.99);
CREATE TABLE Sales_Representatives(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FullName TEXT);
INSERT INTO "Sales_Representatives" VALUES(1,'John Doe');
INSERT INTO "Sales_Representatives" VALUES(2,'Jane Smith');
INSERT INTO "Sales_Representatives" VALUES(3,'Michael Brown');
INSERT INTO "Sales_Representatives" VALUES(4,'Emily White');
INSERT INTO "Sales_Representatives" VALUES(5,'David Black');
CREATE TABLE Sales_Representatives_Territories(
    SalesRepID INTEGER,
    TerritoryID INTEGER,
    FOREIGN KEY(SalesRepID) REFERENCES Sales_Representatives(ID),
    FOREIGN KEY(TerritoryID) REFERENCES Territories(RegionID));
INSERT INTO "Sales_Representatives_Territories" VALUES(1,3);
INSERT INTO "Sales_Representatives_Territories" VALUES(2,3);
INSERT INTO "Sales_Representatives_Territories" VALUES(3,3);
INSERT INTO "Sales_Representatives_Territories" VALUES(4,3);
INSERT INTO "Sales_Representatives_Territories" VALUES(5,3);
INSERT INTO "Sales_Representatives_Territories" VALUES(5,5);
CREATE TABLE Territories(
    RegionID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    OtherColumn1 TEXT,
    OtherColumn2 INT);
INSERT INTO "Territories" VALUES(1,'East Coast','New York',789);
INSERT INTO "Territories" VALUES(2,'West Coast','Los Angeles',362);
INSERT INTO "Territories" VALUES(3,'Central Region','Chicago',785);
INSERT INTO "Territories" VALUES(4,'Northeast','Boston',976);
INSERT INTO "Territories" VALUES(5,'Southwest','Phoenix',218);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Sales_Representatives',5);
INSERT INTO "sqlite_sequence" VALUES('Territories',5);
INSERT INTO "sqlite_sequence" VALUES('Company',5);
INSERT INTO "sqlite_sequence" VALUES('Products',5);
COMMIT;
