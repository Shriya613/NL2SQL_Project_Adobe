BEGIN TRANSACTION;
CREATE TABLE ClientProperties(
    client_id INTEGER,
    property_id INTEGER,
    InterestedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES Clients(client_id),
    FOREIGN KEY (property_id) REFERENCES Properties(property_id));
INSERT INTO "ClientProperties" VALUES(1,1,'2023-01-01 10:00:00');
INSERT INTO "ClientProperties" VALUES(2,2,'2023-02-01 10:00:00');
INSERT INTO "ClientProperties" VALUES(3,3,'2023-03-01 10:00:00');
INSERT INTO "ClientProperties" VALUES(4,4,'2023-04-01 10:00:00');
INSERT INTO "ClientProperties" VALUES(5,5,'2023-05-01 10:00:00');
CREATE TABLE Clients(
    client_id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Email TEXT,
    PhoneNumber TEXT);
INSERT INTO "Clients" VALUES(1,'John Doe','john@example.com','555-111-2222');
INSERT INTO "Clients" VALUES(2,'Jane Smith','janemsmith@email.com','555-333-4444');
INSERT INTO "Clients" VALUES(3,'Alice Johnson','alice.johnson@yahoo.com','555-555-6666');
INSERT INTO "Clients" VALUES(4,'Bob Brown','bobbrown@gmail.com','555-777-8888');
INSERT INTO "Clients" VALUES(5,'Carol White','carolwhite@hotmail.com','555-999-0000');
CREATE TABLE "Properties"(
    property_id INTEGER PRIMARY KEY AUTOINCREMENT,
    Location TEXT,
    Real_Estate_Offices INTEGER,
    Price REAL,
    Property_Type TEXT,
    Square_Feet INTEGER);
INSERT INTO "Properties" VALUES(1,'123 Main St',2,300000.0,'Single Family Home',2000);
INSERT INTO "Properties" VALUES(2,'456 Elm St',1,400000.0,'Condo',1200);
INSERT INTO "Properties" VALUES(3,'789 Oak St',3,500000.0,'Townhouse',2500);
INSERT INTO "Properties" VALUES(4,'999 Pine St',1,600000.0,'Loft',2000);
INSERT INTO "Properties" VALUES(5,'222 Maple Ave',2,450000.0,'Single Family Home',3000);
CREATE TABLE RealEstateOffices(
    real_estate_office_id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT);
INSERT INTO "RealEstateOffices" VALUES(1,'ABC Realty','123 Office Rd','555-123-4567');
INSERT INTO "RealEstateOffices" VALUES(2,'XYZ Real Estate','456 Agency Blvd','555-234-5678');
INSERT INTO "RealEstateOffices" VALUES(3,'Mega Brokers','789 Listing Ln','555-345-6789');
INSERT INTO "RealEstateOffices" VALUES(4,'Local Agents','999 Open House Dr','555-456-7890');
INSERT INTO "RealEstateOffices" VALUES(5,'First Choice Homes','222 Sell Now St','555-567-8901');
CREATE TABLE Transactions(
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    TransactionDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    client_id INTEGER,
    property_id INTEGER,
    TypeOfTransaction TEXT,
    FOREIGN KEY (client_id) REFERENCES Clients(client_id),
    FOREIGN KEY (property_id) REFERENCES Properties(property_id));
INSERT INTO "Transactions" VALUES(1,'2023-01-10 10:00:00',1,1,'Purchase');
INSERT INTO "Transactions" VALUES(2,'2023-02-10 10:00:00',2,2,'Rent');
INSERT INTO "Transactions" VALUES(3,'2023-03-10 10:00:00',3,3,'Lease');
INSERT INTO "Transactions" VALUES(4,'2023-04-10 10:00:00',4,4,'Purchase');
INSERT INTO "Transactions" VALUES(5,'2023-05-10 10:00:00',5,5,'Rent');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('RealEstateOffices',5);
INSERT INTO "sqlite_sequence" VALUES('Clients',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
INSERT INTO "sqlite_sequence" VALUES('Properties',5);
COMMIT;
