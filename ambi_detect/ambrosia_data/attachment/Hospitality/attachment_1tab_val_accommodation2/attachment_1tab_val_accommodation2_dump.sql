BEGIN TRANSACTION;
CREATE TABLE "Accommodations" (
    AccommodationID INTEGER PRIMARY KEY AUTOINCREMENT, RoomNumber TEXT,
    TypeOfAccomodation TEXT ,
    PricePerNight REAL,
    NumberGuests INT,
    HotelID INTEGER,
    FOREIGN KEY (HotelID) REFERENCES Hotels(HotelID));
INSERT INTO "Accommodations" VALUES(1,'101','Family Suite',150.0,4,1);
INSERT INTO "Accommodations" VALUES(2,'102','Family Suite',200.0,4,1);
INSERT INTO "Accommodations" VALUES(3,'201','Single Room',150.0,1,2);
INSERT INTO "Accommodations" VALUES(4,'202','Single Room',200.0,1,2);
INSERT INTO "Accommodations" VALUES(5,'301','Deluxe Double',250.0,2,3);
INSERT INTO "Accommodations" VALUES(6,'401','Standard King',100.0,1,4);
INSERT INTO "Accommodations" VALUES(7,'501','Junior Suite',300.0,3,5);
CREATE TABLE Bookings (
    BookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    CheckInDate DATE,
    CheckOutDate DATE,
    AccommodationID INTEGER,
    GuestID INTEGER,
    TotalCost REAL,
    FOREIGN KEY (AccommodationID) REFERENCES Accommodations(AccommodationID),
    FOREIGN KEY (GuestID) REFERENCES Guests(GuestID));
INSERT INTO "Bookings" VALUES(1,'2022-08-01','2022-08-05',1,1,750.0);
INSERT INTO "Bookings" VALUES(2,'2022-08-10','2022-08-14',2,2,1000.0);
INSERT INTO "Bookings" VALUES(3,'2022-08-15','2022-08-20',3,3,1500.0);
INSERT INTO "Bookings" VALUES(4,'2022-08-21','2022-08-25',4,4,300.0);
INSERT INTO "Bookings" VALUES(5,'2022-08-26','2022-08-30',5,5,1200.0);
CREATE TABLE Guests (
    GuestID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT UNIQUE);
INSERT INTO "Guests" VALUES(1,'John','Doe','john@example.com');
INSERT INTO "Guests" VALUES(2,'Jane','Smith','jane@example.com');
INSERT INTO "Guests" VALUES(3,'Michael','Brown','michael@example.com');
INSERT INTO "Guests" VALUES(4,'Emma','Wilson','emma@example.com');
INSERT INTO "Guests" VALUES(5,'Alice','Jackson','alice@example.com');
CREATE TABLE Hotels (
    HotelID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    City TEXT,
    Country TEXT);
INSERT INTO "Hotels" VALUES(1,'Grand Central','34 Vanderbilt Ave','New York','USA');
INSERT INTO "Hotels" VALUES(2,'Hilton Garden Inn','7982 West Sunrise Blvd.','Plantation','USA');
INSERT INTO "Hotels" VALUES(3,'The Ritz Carlton','160 E 53rd St','New York','USA');
INSERT INTO "Hotels" VALUES(4,'Radisson Blu Mall of America','2100 Killebrew Drive','Bloomington','USA');
INSERT INTO "Hotels" VALUES(5,'Holiday Inn Express','2804 S College Blvd','Fort Collins','USA');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hotels',5);
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
INSERT INTO "sqlite_sequence" VALUES('Accommodations',7);
COMMIT;
