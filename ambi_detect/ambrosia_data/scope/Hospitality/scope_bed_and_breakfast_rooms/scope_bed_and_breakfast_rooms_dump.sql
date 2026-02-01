BEGIN TRANSACTION;
CREATE TABLE Bed_and_breakfast(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    rating REAL  -- assuming ratings from 0-5
);
INSERT INTO "Bed_and_breakfast" VALUES(1,'The Cozy Inn','123 Main St',4.7);
INSERT INTO "Bed_and_breakfast" VALUES(2,'Parkside Lodge','456 Park Ave',4.9);
INSERT INTO "Bed_and_breakfast" VALUES(3,'Seaside Retreat','789 Beach Blvd',4.5);
INSERT INTO "Bed_and_breakfast" VALUES(4,'Mountain Haven','1011 Skyline Dr',4.8);
INSERT INTO "Bed_and_breakfast" VALUES(5,'Uptown Manor','1314 City Center Dr',4.6);
CREATE TABLE Bed_and_breakfast_rooms(
    bed_id INTEGER,
    room_id INTEGER,
    FOREIGN KEY (bed_id) REFERENCES Bed_and_breakfast(id),
    FOREIGN KEY (room_id) REFERENCES Rooms(room_type)
);
INSERT INTO "Bed_and_breakfast_rooms" VALUES(1,'Single Room');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(2,'Double Room');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(3,'Suite');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(4,'Deluxe Suite');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(5,'Family Room');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(1,'Double Room');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(2,'Family Room');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(3,'Double Room');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(4,'Double Room');
INSERT INTO "Bed_and_breakfast_rooms" VALUES(5,'Double Room');
CREATE TABLE Bookings(
    booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bed_id INTEGER,
    customer_id INTEGER,
    checkin DATE,
    checkout DATE,
    total_price DECIMAL(10,2),
    FOREIGN KEY (bed_id) REFERENCES Bed_and_breakfast_rooms(bed_id),
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
INSERT INTO "Bookings" VALUES(1,1,1,'2022-01-01','2022-01-03',250);
INSERT INTO "Bookings" VALUES(2,2,2,'2022-01-04','2022-01-06',350);
INSERT INTO "Bookings" VALUES(3,3,3,'2022-01-07','2022-01-09',300);
INSERT INTO "Bookings" VALUES(4,4,4,'2022-01-10','2022-01-12',400);
INSERT INTO "Bookings" VALUES(5,5,5,'2022-01-13','2022-01-15',350);
CREATE TABLE Customers(
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com','555-1234');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com','555-5678');
INSERT INTO "Customers" VALUES(3,'Alice','Wonderland','alice.wonderland@example.com','555-9012');
INSERT INTO "Customers" VALUES(4,'Charlie','Brown','charlie.brown@example.com','555-0123');
INSERT INTO "Customers" VALUES(5,'David','White','david.white@example.com','555-1357');
CREATE TABLE Rooms(
    room_type TEXT PRIMARY KEY,
    price DECIMAL(10,2) -- Assuming prices have maximum 8 digits before decimal point and 2 after decimal
);
INSERT INTO "Rooms" VALUES('Single Room',100);
INSERT INTO "Rooms" VALUES('Double Room',150);
INSERT INTO "Rooms" VALUES('Suite',250);
INSERT INTO "Rooms" VALUES('Deluxe Suite',300);
INSERT INTO "Rooms" VALUES('Family Room',200);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Bed_and_breakfast',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
COMMIT;
