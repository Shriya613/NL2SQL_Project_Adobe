BEGIN TRANSACTION;
CREATE TABLE Bookings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guestId INTEGER,
    chaletId INTEGER,
    bookingDate DATETIME,
    totalCost REAL,
    FOREIGN KEY(guestId) REFERENCES Guests(id),
    FOREIGN KEY(chaletId) REFERENCES Chalets(id)
);
INSERT INTO "Bookings" VALUES(1,1,1,'2022-01-01 00:00:00',1000.0);
INSERT INTO "Bookings" VALUES(2,2,2,'2022-01-01 00:00:00',1200.0);
INSERT INTO "Bookings" VALUES(3,3,3,'2022-01-01 00:00:00',1500.0);
INSERT INTO "Bookings" VALUES(4,4,4,'2022-01-01 00:00:00',1800.0);
INSERT INTO "Bookings" VALUES(5,5,5,'2022-01-01 00:00:00',2000.0);
CREATE TABLE Cabins(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cabinName TEXT,
    pricePerNight REAL ,
    location TEXT,
    capacity INTEGER
);
INSERT INTO "Cabins" VALUES(1,'Lakeshore',200.0,'Lake',4);
INSERT INTO "Cabins" VALUES(2,'Riverview',300.0,'Riverside',6);
INSERT INTO "Cabins" VALUES(3,'Seaside',250.0,'Coast',8);
INSERT INTO "Cabins" VALUES(4,'Woodside',150.0,'Forest',4);
INSERT INTO "Cabins" VALUES(5,'Valley',275.0,'Hills',6);
CREATE TABLE Chalets(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chaletName TEXT,
    pricePerNight REAL ,
    location TEXT,
    capacity INTEGER
);
INSERT INTO "Chalets" VALUES(1,'Aurora',200.0,'Alps',6);
INSERT INTO "Chalets" VALUES(2,'Sunshine',300.0,'Beach',8);
INSERT INTO "Chalets" VALUES(3,'Moonlight',250.0,'City',10);
INSERT INTO "Chalets" VALUES(4,'Starlight',150.0,'Forest',8);
INSERT INTO "Chalets" VALUES(5,'Dawn',275.0,'Mountain',6);
CREATE TABLE Guests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT,
    email TEXT,
    phoneNumber VARCHAR(15),
    checkIn DATETIME,
    checkOut DATETIME
);
INSERT INTO "Guests" VALUES(1,'John','Doe','john@gmail.com','1234567890','2022-01-01 10:00:00','2022-01-05 15:00:00');
INSERT INTO "Guests" VALUES(2,'Jane','Smith','jane@gmail.com','0987654321','2022-01-01 15:00:00','2022-01-05 10:00:00');
INSERT INTO "Guests" VALUES(3,'Jack','Brown','jack@gmail.com','1112223334','2022-01-01 12:00:00','2022-01-05 11:00:00');
INSERT INTO "Guests" VALUES(4,'Samantha','White','samantha@gmail.com','4321123456','2022-01-01 13:00:00','2022-01-05 12:00:00');
INSERT INTO "Guests" VALUES(5,'Emma','Black','emma@gmail.com','3456789123','2022-01-01 14:00:00','2022-01-05 13:00:00');
CREATE TABLE Reviews(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guestId INTEGER,
    chaletId INTEGER,
    rating INTEGER ,
    comment TEXT,
    reviewDate DATETIME,
    FOREIGN KEY(guestId) REFERENCES Guests(id),
    FOREIGN KEY(chaletId) REFERENCES Chalets(id)
);
INSERT INTO "Reviews" VALUES(1,1,1,5,'Great stay!','2022-01-01 00:00:00');
INSERT INTO "Reviews" VALUES(2,2,2,4,'Good experience!','2022-01-01 00:00:00');
INSERT INTO "Reviews" VALUES(3,3,3,3,'Fine, could have been better.','2022-01-01 00:00:00');
INSERT INTO "Reviews" VALUES(4,4,4,5,'Excellent service!','2022-01-01 00:00:00');
INSERT INTO "Reviews" VALUES(5,5,5,4,'Good value for money.','2022-01-01 00:00:00');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Chalets',5);
INSERT INTO "sqlite_sequence" VALUES('Cabins',5);
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
INSERT INTO "sqlite_sequence" VALUES('Reviews',5);
COMMIT;
