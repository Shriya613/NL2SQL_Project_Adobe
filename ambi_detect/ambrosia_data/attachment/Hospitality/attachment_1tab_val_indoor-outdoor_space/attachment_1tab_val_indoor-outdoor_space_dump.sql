BEGIN TRANSACTION;
CREATE TABLE Amenities(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amenityName TEXT,
    available BOOL DEFAULT FALSE
);
INSERT INTO "Amenities" VALUES(1,'WiFi',1);
INSERT INTO "Amenities" VALUES(2,'Parking Space',1);
INSERT INTO "Amenities" VALUES(3,'Air Conditioner',1);
INSERT INTO "Amenities" VALUES(4,'Projector',0);
CREATE TABLE Booking(
    bookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    roomId INTEGER REFERENCES Room(id),
    customerId INTEGER REFERENCES Customer(customerID),
    startDate DATE,
    endDate DATE,
    totalPrice DECIMAL(10,2)
);
INSERT INTO "Booking" VALUES(1,1,1,'2022-01-10','2022-01-12',2000);
INSERT INTO "Booking" VALUES(2,2,2,'2022-01-15','2022-01-17',3000);
INSERT INTO "Booking" VALUES(3,3,3,'2022-01-20','2022-01-22',4000);
INSERT INTO "Booking" VALUES(4,4,4,'2022-01-25','2022-01-27',5000);
CREATE TABLE Customer(
    customerID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    phoneNumber CHAR(10)
);
INSERT INTO "Customer" VALUES(1,'John','Doe','john@example.com','+123456789');
INSERT INTO "Customer" VALUES(2,'Jane','Smith','jane@example.com','+234567890');
INSERT INTO "Customer" VALUES(3,'Michael','Johnson','michael@example.com','+345678901');
INSERT INTO "Customer" VALUES(4,'Emily','Williams','emily@example.com','+45678902');
CREATE TABLE Room(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Indoor_Outdoor_Space TEXT,
    Size INT,
    Price DECIMAL(10,2)
);
INSERT INTO "Room" VALUES(1,'Conference Hall','Lobby',500,678.99);
INSERT INTO "Room" VALUES(2,'Banquet Hall','Lobby',600,830.99);
INSERT INTO "Room" VALUES(3,'Rooftop Terrace','Terrace',500,1000.99);
INSERT INTO "Room" VALUES(4,'Garden Patio','Terrace',400,870.99);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Room',4);
INSERT INTO "sqlite_sequence" VALUES('Amenities',4);
INSERT INTO "sqlite_sequence" VALUES('Booking',4);
INSERT INTO "sqlite_sequence" VALUES('Customer',4);
COMMIT;
