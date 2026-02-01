BEGIN TRANSACTION;
CREATE TABLE GuestHouseSuites(
    guesthouseID INTEGER,
    suiteType TEXT,
    FOREIGN KEY(guesthouseID) REFERENCES GuestHouses(houseID),
    FOREIGN KEY(suiteType) REFERENCES Suites(suiteType));
INSERT INTO "GuestHouseSuites" VALUES(1,'Standard Suite');
INSERT INTO "GuestHouseSuites" VALUES(1,'Family Suite');
INSERT INTO "GuestHouseSuites" VALUES(2,'Deluxe Suite');
INSERT INTO "GuestHouseSuites" VALUES(2,'Family Suite');
INSERT INTO "GuestHouseSuites" VALUES(3,'Standard Suite');
INSERT INTO "GuestHouseSuites" VALUES(3,'Family Suite');
INSERT INTO "GuestHouseSuites" VALUES(4,'Deluxe Suite');
INSERT INTO "GuestHouseSuites" VALUES(4,'Executive Suite');
INSERT INTO "GuestHouseSuites" VALUES(5,'Deluxe Suite');
INSERT INTO "GuestHouseSuites" VALUES(5,'Family Suite');
INSERT INTO "GuestHouseSuites" VALUES(3,'Deluxe Suite');
INSERT INTO "GuestHouseSuites" VALUES(4,'Family Suite');
INSERT INTO "GuestHouseSuites" VALUES(5,'Standard Suite');
CREATE TABLE GuestHouses(
    houseID INTEGER PRIMARY KEY AUTOINCREMENT,
    houseName TEXT,
    location TEXT,
    capacity INTEGER,
    pricePerNight REAL);
INSERT INTO "GuestHouses" VALUES(1,'Cozy Inn','New York City',10,150.0);
INSERT INTO "GuestHouses" VALUES(2,'Urban Retreat','Los Angeles',15,200.0);
INSERT INTO "GuestHouses" VALUES(3,'Mountain Lodge','Denver',8,180.0);
INSERT INTO "GuestHouses" VALUES(4,'Beachfront Paradise','Miami Beach',12,250.0);
INSERT INTO "GuestHouses" VALUES(5,'City Center Suite','Chicago',6,175.0);
CREATE TABLE Guests(
    guestID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    phoneNumber TEXT);
INSERT INTO "Guests" VALUES(1,'John','Doe','john@example.com','+1-555-123-4567');
INSERT INTO "Guests" VALUES(2,'Jane','Smith','jane@example.com','+1-555-987-6543');
INSERT INTO "Guests" VALUES(3,'Michael','Brown','michael@example.com','+1-555-321-9876');
INSERT INTO "Guests" VALUES(4,'Emily','White','emily@example.com','+1-555-135-7890');
INSERT INTO "Guests" VALUES(5,'David','Black','david@example.com','+1-555-456-1234');
CREATE TABLE RoomAmenities(
    amenID INTEGER PRIMARY KEY AUTOINCREMENT,
    amenName TEXT,
    amenCost REAL);
INSERT INTO "RoomAmenities" VALUES(1,'Free WiFi',0.0);
INSERT INTO "RoomAmenities" VALUES(2,'Parking',10.99);
INSERT INTO "RoomAmenities" VALUES(3,'Gym Access',2.5);
INSERT INTO "RoomAmenities" VALUES(4,'Breakfast Included',8.75);
INSERT INTO "RoomAmenities" VALUES(5,'Laundry Service',3.0);
CREATE TABLE Suites(
    suiteType TEXT PRIMARY KEY);
INSERT INTO "Suites" VALUES('Standard Suite');
INSERT INTO "Suites" VALUES('Deluxe Suite');
INSERT INTO "Suites" VALUES('Executive Suite');
INSERT INTO "Suites" VALUES('Family Suite');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('RoomAmenities',5);
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('GuestHouses',5);
COMMIT;
