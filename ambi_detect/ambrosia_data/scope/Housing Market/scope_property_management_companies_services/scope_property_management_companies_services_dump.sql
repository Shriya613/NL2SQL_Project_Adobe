BEGIN TRANSACTION;
CREATE TABLE Clients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_name TEXT,
    contact TEXT
);
INSERT INTO "Clients" VALUES(1,'Client1 Name','Contact1');
INSERT INTO "Clients" VALUES(2,'Client2 Name','Contact2');
INSERT INTO "Clients" VALUES(3,'Client3 Name','Contact3');
INSERT INTO "Clients" VALUES(4,'Client4 Name','Contact4');
INSERT INTO "Clients" VALUES(5,'Client5 Name','Contact5');
CREATE TABLE Properties(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_address TEXT,
    price REAL 
);
INSERT INTO "Properties" VALUES(1,'Property1 Address',1000000.0);
INSERT INTO "Properties" VALUES(2,'Property2 Address',800000.0);
INSERT INTO "Properties" VALUES(3,'Property3 Address',900000.0);
INSERT INTO "Properties" VALUES(4,'Property4 Address',700000.0);
INSERT INTO "Properties" VALUES(5,'Property5 Address',600000.0);
CREATE TABLE Property_Management_Companies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT
);
INSERT INTO "Property_Management_Companies" VALUES(1,'Company1','Address1');
INSERT INTO "Property_Management_Companies" VALUES(2,'Company2','Address2');
INSERT INTO "Property_Management_Companies" VALUES(3,'Company3','Address3');
INSERT INTO "Property_Management_Companies" VALUES(4,'Company4','Address4');
INSERT INTO "Property_Management_Companies" VALUES(5,'Company5','Address5');
CREATE TABLE Property_Management_Companies_Services(
    company_id INTEGER,
    service_id INTEGER,
    FOREIGN KEY(company_id) REFERENCES Property_Management_Companies(id),
    FOREIGN KEY(service_id) REFERENCES Services(id),
    PRIMARY KEY(company_id, service_id)
);
INSERT INTO "Property_Management_Companies_Services" VALUES(1,1);
INSERT INTO "Property_Management_Companies_Services" VALUES(2,1);
INSERT INTO "Property_Management_Companies_Services" VALUES(3,1);
INSERT INTO "Property_Management_Companies_Services" VALUES(4,1);
INSERT INTO "Property_Management_Companies_Services" VALUES(5,1);
INSERT INTO "Property_Management_Companies_Services" VALUES(1,2);
INSERT INTO "Property_Management_Companies_Services" VALUES(2,2);
INSERT INTO "Property_Management_Companies_Services" VALUES(3,2);
INSERT INTO "Property_Management_Companies_Services" VALUES(4,2);
INSERT INTO "Property_Management_Companies_Services" VALUES(5,2);
CREATE TABLE Services(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT
);
INSERT INTO "Services" VALUES(1,'Lease Agreement');
INSERT INTO "Services" VALUES(2,'Rent Collection');
INSERT INTO "Services" VALUES(3,'Maintenance');
INSERT INTO "Services" VALUES(4,'Eviction Service');
INSERT INTO "Services" VALUES(5,'Tenant Screening');
CREATE TABLE Transactions(
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id INTEGER,
    client_id INTEGER,
    management_company_id INTEGER,
    service_id INTEGER,
    FOREIGN KEY(property_id) REFERENCES Properties(id),
    FOREIGN KEY(client_id) REFERENCES Clients(id),
    FOREIGN KEY(management_company_id) REFERENCES Property_Management_Companies(id),
    FOREIGN KEY(service_id) REFERENCES Services(id)
);
INSERT INTO "Transactions" VALUES(1,1,1,1,1);
INSERT INTO "Transactions" VALUES(2,2,2,2,2);
INSERT INTO "Transactions" VALUES(3,3,3,3,3);
INSERT INTO "Transactions" VALUES(4,4,4,4,4);
INSERT INTO "Transactions" VALUES(5,5,5,5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Property_Management_Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Services',5);
INSERT INTO "sqlite_sequence" VALUES('Properties',5);
INSERT INTO "sqlite_sequence" VALUES('Clients',5);
INSERT INTO "sqlite_sequence" VALUES('Transactions',5);
COMMIT;
