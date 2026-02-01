BEGIN TRANSACTION;
CREATE TABLE Bookings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkInDate DATE,
    checkOutDate DATE,
    totalPrice REAL,
    roomId INTEGER,
    FOREIGN KEY(roomId) REFERENCES Rooms(id)
);
INSERT INTO "Bookings" VALUES(1,'2022-01-01','2022-01-07',700.0,1);
INSERT INTO "Bookings" VALUES(2,'2022-01-15','2022-01-21',1000.0,2);
INSERT INTO "Bookings" VALUES(3,'2022-02-01','2022-02-07',400.0,3);
INSERT INTO "Bookings" VALUES(4,'2022-02-15','2022-02-21',600.0,4);
INSERT INTO "Bookings" VALUES(5,'2022-03-01','2022-03-07',800.0,5);
CREATE TABLE Customer_Bookings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customerId INTEGER,
    bookingId INTEGER,
    FOREIGN KEY(customerId) REFERENCES Customers(id),
    FOREIGN KEY(bookingId) REFERENCES Bookings(id)
);
INSERT INTO "Customer_Bookings" VALUES(1,1,1);
INSERT INTO "Customer_Bookings" VALUES(2,2,2);
INSERT INTO "Customer_Bookings" VALUES(3,3,3);
INSERT INTO "Customer_Bookings" VALUES(4,4,4);
INSERT INTO "Customer_Bookings" VALUES(5,5,5);
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phoneNumber TEXT,
    Vacation_Option TEXT ,
    Activities TEXT 
);
INSERT INTO "Customers" VALUES(1,'John Doe','john@example.com','1234567890','Cruise Ship','Dance Class');
INSERT INTO "Customers" VALUES(2,'Jane Smith','jane@example.com','0987654321','Cruise Ship','Yoga');
INSERT INTO "Customers" VALUES(3,'Michael Johnson','michael@example.com','5555555555','All-Inclusive Resort','Dance Class');
INSERT INTO "Customers" VALUES(4,'Emily Davis','emily@example.com','4444444444','All-Inclusive Resort','Hiking');
INSERT INTO "Customers" VALUES(5,'Alice Brown','alice@example.com','3333333333','Cruise Ship','Snorkeling');
INSERT INTO "Customers" VALUES(6,'Bob Wilson','bob@example.com','2222222222','All-Inclusive Resort','Golf');
CREATE TABLE Rooms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    capacity INTEGER,
    price REAL
);
INSERT INTO "Rooms" VALUES(1,'Single Room',1,100.0);
INSERT INTO "Rooms" VALUES(2,'Double Room',2,200.0);
INSERT INTO "Rooms" VALUES(3,'Triple Room',3,300.0);
INSERT INTO "Rooms" VALUES(4,'Family Suite',4,400.0);
INSERT INTO "Rooms" VALUES(5,'Luxury Suite',2,500.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',6);
INSERT INTO "sqlite_sequence" VALUES('Rooms',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
INSERT INTO "sqlite_sequence" VALUES('Customer_Bookings',5);
COMMIT;
