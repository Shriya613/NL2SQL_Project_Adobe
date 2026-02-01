BEGIN TRANSACTION;
CREATE TABLE Apartment_Buildings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    city TEXT,
    state TEXT,
    zipcode INTEGER,
    num_units INTEGER);
INSERT INTO "Apartment_Buildings" VALUES(1,'Building A','10 Main St','New York','NY',10001,30);
INSERT INTO "Apartment_Buildings" VALUES(2,'Building B','20 Elm St','Los Angeles','CA',90001,40);
INSERT INTO "Apartment_Buildings" VALUES(3,'Building C','30 Oak St','Chicago','IL',60601,50);
INSERT INTO "Apartment_Buildings" VALUES(4,'Building D','40 Pine St','Miami','FL',33101,60);
INSERT INTO "Apartment_Buildings" VALUES(5,'Building E','50 Maple Ave','Dallas','TX',75201,70);
CREATE TABLE Apartment_Buildings_Units(
    building_id INTEGER,
    unit_id INTEGER,
    FOREIGN KEY (building_id) REFERENCES Apartment_Buildings(id),
    FOREIGN KEY (unit_id) REFERENCES Units(id));
INSERT INTO "Apartment_Buildings_Units" VALUES(1,1);
INSERT INTO "Apartment_Buildings_Units" VALUES(2,1);
INSERT INTO "Apartment_Buildings_Units" VALUES(3,1);
INSERT INTO "Apartment_Buildings_Units" VALUES(4,1);
INSERT INTO "Apartment_Buildings_Units" VALUES(5,1);
INSERT INTO "Apartment_Buildings_Units" VALUES(5,4);
CREATE TABLE RentPayments(
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount DECIMAL(8,2) ,
    paid_date DATE,
    due_date DATE,
    tenant_id INTEGER,
    FOREIGN KEY (tenant_id) REFERENCES Tenants(tenant_id));
INSERT INTO "RentPayments" VALUES(1,1200,'2021-10-01','2021-10-01',1);
INSERT INTO "RentPayments" VALUES(2,1200,'2021-10-01','2021-10-01',2);
INSERT INTO "RentPayments" VALUES(3,1200,'2021-10-01','2021-10-01',3);
INSERT INTO "RentPayments" VALUES(4,1200,'2021-10-01','2021-10-01',4);
INSERT INTO "RentPayments" VALUES(5,1200,'2021-10-01','2021-10-01',5);
CREATE TABLE Tenants(
    tenant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT,
    email TEXT,
    phone TEXT);
INSERT INTO "Tenants" VALUES(1,'John','Smith','johnsmith@email.com','555-123-4567');
INSERT INTO "Tenants" VALUES(2,'Jane','Doe','janedoemail@email.com','555-987-6543');
INSERT INTO "Tenants" VALUES(3,'Alice','Wilson','alicewilson@email.com','555-321-9876');
INSERT INTO "Tenants" VALUES(4,'Bob','Jackson','bobjackson@email.com','555-654-3210');
INSERT INTO "Tenants" VALUES(5,'Emily','Martin','emilymartin@email.com','555-135-7890');
CREATE TABLE Units(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_type TEXT,
    description TEXT);
INSERT INTO "Units" VALUES(1,'Studio_Apartment','Spacious studio apartment with modern amenities');
INSERT INTO "Units" VALUES(2,'One-Bedroom','Comfortable one-bedroom apartment with a fully equipped kitchen');
INSERT INTO "Units" VALUES(3,'Two-Bedroom','Beautiful two-bedroom apartment perfect for families');
INSERT INTO "Units" VALUES(4,'Three-Bedroom','Luxurious three-bedroom apartment with stunning views');
INSERT INTO "Units" VALUES(5,'Four-Bedroom','Elegant four-bedroom apartment with private balcony');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Apartment_Buildings',5);
INSERT INTO "sqlite_sequence" VALUES('Units',5);
INSERT INTO "sqlite_sequence" VALUES('Tenants',5);
INSERT INTO "sqlite_sequence" VALUES('RentPayments',5);
COMMIT;
