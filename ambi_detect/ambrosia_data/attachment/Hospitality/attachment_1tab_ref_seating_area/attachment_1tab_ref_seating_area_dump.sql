BEGIN TRANSACTION;
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    PhoneNumber TEXT,
    EmailAddress TEXT UNIQUE
, FullName TEXT);
INSERT INTO "Customers" VALUES(1,'+1234567890','john.doe@example.com','John Doe');
INSERT INTO "Customers" VALUES(2,'+0987654321','jane.smith@example.com','Jane Smith');
INSERT INTO "Customers" VALUES(3,'+1123456789','alice.brown@example.com','Alice Brown');
INSERT INTO "Customers" VALUES(4,'+0987654320','charlie.black@example.com','Charlie Black');
INSERT INTO "Customers" VALUES(5,'+1345678901','david.white@example.com','David White');
CREATE TABLE FloorPlan(
    FloorplanID INTEGER PRIMARY KEY AUTOINCREMENT,
    NumberOfSeats INT,
    TableId INT REFERENCES Tables(TableID),
    FOREIGN KEY (TableId) REFERENCES Tables(TableID)
);
INSERT INTO "FloorPlan" VALUES(1,50,1);
INSERT INTO "FloorPlan" VALUES(2,40,2);
INSERT INTO "FloorPlan" VALUES(3,50,3);
INSERT INTO "FloorPlan" VALUES(4,40,4);
INSERT INTO "FloorPlan" VALUES(5,60,5);
CREATE TABLE FoodOrders(
    OrderID INTEGER PRIMARY KEY AUTOINCREMENT,
    ReservationID INTEGER,
    MenuItem ID TEXT,
    Quantity INT,
    SpecialInstructions TEXT,
    TotalPrice DECIMAL(10,2),
    PaymentStatus TEXT ,
    FOREIGN KEY (ReservationID) REFERENCES Reservations(ReservationID)
);
INSERT INTO "FoodOrders" VALUES(1,1,'Steak',2,'',75,'Paid');
INSERT INTO "FoodOrders" VALUES(2,2,'Salmon',3,'',90,'Paid');
INSERT INTO "FoodOrders" VALUES(3,3,'Veggie Burger',2,'No cheese',15,'Paid');
INSERT INTO "FoodOrders" VALUES(4,4,'Chicken Caesar Salad',1,'',18,'Paid');
INSERT INTO "FoodOrders" VALUES(5,5,'Pasta Primavera',1,'Gluten-free',20,'Paid');
CREATE TABLE Reservations(
    ReservationID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    ReservedDate DATETIME,
    ReservedTime TIME,
    NumGuests INT,
    TableId INT REFERENCES Tables(TableID),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (TableId) REFERENCES Tables(TableID)
);
INSERT INTO "Reservations" VALUES(1,1,'2023-01-01 18:00','19:00',2,1);
INSERT INTO "Reservations" VALUES(2,2,'2023-01-02 19:00','20:00',4,2);
INSERT INTO "Reservations" VALUES(3,3,'2023-01-03 20:00','21:00',3,3);
INSERT INTO "Reservations" VALUES(4,4,'2023-01-04 21:00','22:00',5,4);
INSERT INTO "Reservations" VALUES(5,5,'2023-01-05 22:00','23:00',6,5);
CREATE TABLE Tables(
    TableID INTEGER PRIMARY KEY AUTOINCREMENT,
    SeatingArea TEXT,
    Capacity INT
);
INSERT INTO "Tables" VALUES(1,'Patio',50);
INSERT INTO "Tables" VALUES(2,'Patio',40);
INSERT INTO "Tables" VALUES(3,'Veranda',50);
INSERT INTO "Tables" VALUES(4,'Veranda',40);
INSERT INTO "Tables" VALUES(5,'Indoor',60);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Tables',5);
INSERT INTO "sqlite_sequence" VALUES('FloorPlan',5);
INSERT INTO "sqlite_sequence" VALUES('Reservations',5);
INSERT INTO "sqlite_sequence" VALUES('FoodOrders',5);
COMMIT;
