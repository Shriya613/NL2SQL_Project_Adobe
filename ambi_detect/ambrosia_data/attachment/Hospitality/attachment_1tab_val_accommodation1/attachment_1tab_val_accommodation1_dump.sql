BEGIN TRANSACTION;
CREATE TABLE Accommodations (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Type_of_Accommodation TEXT ,
    Price REAL
);
INSERT INTO "Accommodations" VALUES(1,'Grand Hotel','Hotel',50.0);
INSERT INTO "Accommodations" VALUES(2,'Cozy Inn','Hotel',80.0);
INSERT INTO "Accommodations" VALUES(3,'Backpacker Lodge','Youth Hostel',50.0);
INSERT INTO "Accommodations" VALUES(4,'World Explorer Hostel','Youth Hostel',60.0);
INSERT INTO "Accommodations" VALUES(5,'Paradise Resort','Resort',120.0);
INSERT INTO "Accommodations" VALUES(6,'Luxury Suites','Apartment',130.0);
CREATE TABLE Booking (
    BookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    AccommodationID INTEGER,
    CheckInDate DATE,
    CheckOutDate DATE,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (AccommodationID) REFERENCES Accommodations(ID)
);
INSERT INTO "Booking" VALUES(1,1,1,'2022-07-10','2022-07-17');
INSERT INTO "Booking" VALUES(2,2,2,'2022-08-05','2022-08-12');
INSERT INTO "Booking" VALUES(3,3,3,'2022-09-01','2022-09-08');
INSERT INTO "Booking" VALUES(4,4,4,'2022-10-15','2022-10-22');
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Customers" VALUES(3,'Alice','Wonderland','alice.wonderland@example.com');
INSERT INTO "Customers" VALUES(4,'Bob','Builder','bob.builder@example.com');
CREATE TABLE Reviews (
    ReviewID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccommodationID INTEGER,
    Rating INTEGER ,
    Comment TEXT,
    FOREIGN KEY (AccommodationID) REFERENCES Accommodations(ID)
);
INSERT INTO "Reviews" VALUES(1,1,4,'Great location but noisy.');
INSERT INTO "Reviews" VALUES(2,2,5,'Very comfortable and clean.');
INSERT INTO "Reviews" VALUES(3,3,3,'Nice place with friendly staff.');
INSERT INTO "Reviews" VALUES(4,4,2,'Not what I expected.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Accommodations',6);
INSERT INTO "sqlite_sequence" VALUES('Customers',4);
INSERT INTO "sqlite_sequence" VALUES('Booking',4);
INSERT INTO "sqlite_sequence" VALUES('Reviews',4);
COMMIT;
