BEGIN TRANSACTION;
CREATE TABLE House (
    id INTEGER PRIMARY KEY,
    property_type_id INTEGER,
    neighborhood_id INTEGER,
    address VARCHAR(255),
    square_footage INTEGER,
    bedrooms INTEGER,
    bathrooms INTEGER,
    "Extras" DECIMAL(10,2),
    House_Price DECIMAL(10,2),
    FOREIGN KEY (property_type_id) REFERENCES PropertyType(id),
    FOREIGN KEY (neighborhood_id) REFERENCES Neighborhood(id)
);
INSERT INTO "House" VALUES(1,1,1,'123 Main St',2000,4,3,150,700000);
INSERT INTO "House" VALUES(2,2,2,'456 Elm St',1500,3,2,100,600000);
INSERT INTO "House" VALUES(3,3,3,'789 Oak St',1000,2,1,80,350000);
INSERT INTO "House" VALUES(4,4,4,'101 Pine St',2500,4,3,200,950000);
INSERT INTO "House" VALUES(5,5,5,'202 Maple Ave',1200,3,2,120,400000);
CREATE TABLE Location (
    id INTEGER PRIMARY KEY,
    city VARCHAR(255),
    state VARCHAR(255),
    country VARCHAR(255)
);
INSERT INTO "Location" VALUES(1,'New York','NY','USA');
INSERT INTO "Location" VALUES(2,'Los Angeles','CA','USA');
INSERT INTO "Location" VALUES(3,'Chicago','IL','USA');
INSERT INTO "Location" VALUES(4,'Miami','FL','USA');
INSERT INTO "Location" VALUES(5,'Boston','MA','USA');
CREATE TABLE Mortgage (
    id INTEGER PRIMARY KEY,
    sale_record_id INTEGER,
    loan_amount DECIMAL(10,2),
    interest_rate FLOAT,
    mortgage_term INTEGER,
    monthly_payment DECIMAL(10,2),
    start_date DATE DEFAULT CURRENT_DATE,
    end_date DATE,
    FOREIGN KEY (sale_record_id) REFERENCES SaleRecord(id)
);
INSERT INTO "Mortgage" VALUES(1,1,800000,3.5,30,2500,'2023-01-01','2033-01-01');
INSERT INTO "Mortgage" VALUES(2,2,500000,3.0,30,2100,'2023-01-02','2053-01-02');
INSERT INTO "Mortgage" VALUES(3,3,250000,4.0,30,1000,'2023-01-03','2053-01-03');
INSERT INTO "Mortgage" VALUES(4,4,900000,3.25,30,2900,'2023-01-04','2053-01-04');
INSERT INTO "Mortgage" VALUES(5,5,400000,4.5,30,1500,'2023-01-05','2053-01-05');
CREATE TABLE Neighborhood (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    location_id INTEGER,
    FOREIGN KEY (location_id) REFERENCES Location(id)
);
INSERT INTO "Neighborhood" VALUES(1,'Downtown',1);
INSERT INTO "Neighborhood" VALUES(2,'Southside',1);
INSERT INTO "Neighborhood" VALUES(3,'Westside',2);
INSERT INTO "Neighborhood" VALUES(4,'Northside',3);
INSERT INTO "Neighborhood" VALUES(5,'Eastside',4);
CREATE TABLE PropertyType (
    id INTEGER PRIMARY KEY,
    type VARCHAR(255)
);
INSERT INTO "PropertyType" VALUES(1,'Single Family Home');
INSERT INTO "PropertyType" VALUES(2,'Condo');
INSERT INTO "PropertyType" VALUES(3,'Townhome');
INSERT INTO "PropertyType" VALUES(4,'Apartment');
INSERT INTO "PropertyType" VALUES(5,'Mobile Home');
CREATE TABLE SaleRecord (
    id INTEGER PRIMARY KEY,
    house_id INTEGER,
    sale_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    purchase_price DECIMAL(10,2),
    FOREIGN KEY (house_id) REFERENCES House(id)
);
INSERT INTO "SaleRecord" VALUES(1,1,'2023-01-01 10:00:00',700000);
INSERT INTO "SaleRecord" VALUES(2,2,'2023-01-02 10:00:00',600000);
INSERT INTO "SaleRecord" VALUES(3,3,'2023-01-03 10:00:00',350000);
INSERT INTO "SaleRecord" VALUES(4,4,'2023-01-04 10:00:00',950000);
INSERT INTO "SaleRecord" VALUES(5,5,'2023-01-05 10:00:00',400000);
COMMIT;
