BEGIN TRANSACTION;
CREATE TABLE BookedServices (
    BookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    ReservationID INTEGER,
    ServiceID INTEGER,
    Quantity INTEGER,
    TotalAmount REAL,
    FOREIGN KEY(ReservationID) REFERENCES Reservations(ReservationID),
    FOREIGN KEY(ServiceID) REFERENCES Services(ServiceID)
);
INSERT INTO "BookedServices" VALUES(1,1,1,1,150.0);
INSERT INTO "BookedServices" VALUES(2,1,2,1,50.0);
INSERT INTO "BookedServices" VALUES(3,1,3,1,100.0);
INSERT INTO "BookedServices" VALUES(4,1,4,1,30.0);
INSERT INTO "BookedServices" VALUES(5,1,5,1,15.0);
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com','+123456789');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com','+987654321');
INSERT INTO "Customers" VALUES(3,'Michael','Brown','michael.brown@example.com','+111222333');
INSERT INTO "Customers" VALUES(4,'Emily','White','emily.white@example.com','+444555666');
INSERT INTO "Customers" VALUES(5,'David','Black','david.black@example.com','+777888999');
CREATE TABLE Reservations (
    ReservationID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    CheckInDate TEXT,
    CheckOutDate TEXT,
    TotalAmount REAL,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID)
);
INSERT INTO "Reservations" VALUES(1,1,'2023-01-01','2023-01-05',1000.0);
INSERT INTO "Reservations" VALUES(2,2,'2023-01-10','2023-01-15',1200.0);
INSERT INTO "Reservations" VALUES(3,3,'2023-01-20','2023-01-25',1400.0);
INSERT INTO "Reservations" VALUES(4,4,'2023-02-01','2023-02-05',1600.0);
INSERT INTO "Reservations" VALUES(5,5,'2023-02-10','2023-02-15',1800.0);
CREATE TABLE "Rooms" (
    RoomID INTEGER PRIMARY KEY,
    Location TEXT,
    Spa BOOLEAN DEFAULT FALSE,
    Occupied BOOLEAN DEFAULT FALSE,
    Rate REAL
);
INSERT INTO "Rooms" VALUES(1,'Building A',1,0,299.99);
INSERT INTO "Rooms" VALUES(2,'Building B',0,1,399.99);
INSERT INTO "Rooms" VALUES(3,'Building C',1,1,499.99);
INSERT INTO "Rooms" VALUES(4,'Building D',0,0,599.99);
INSERT INTO "Rooms" VALUES(5,'Building E',1,0,699.99);
CREATE TABLE Services (
    ServiceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    Price REAL
);
INSERT INTO "Services" VALUES(1,'Room upgrade to suite',150.0);
INSERT INTO "Services" VALUES(2,'Extra bed',50.0);
INSERT INTO "Services" VALUES(3,'Late checkout',100.0);
INSERT INTO "Services" VALUES(4,'Breakfast buffet',30.0);
INSERT INTO "Services" VALUES(5,'Welcome drink at bar',15.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Reservations',5);
INSERT INTO "sqlite_sequence" VALUES('Services',5);
INSERT INTO "sqlite_sequence" VALUES('BookedServices',5);
COMMIT;
