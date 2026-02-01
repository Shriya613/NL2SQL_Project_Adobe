BEGIN TRANSACTION;
CREATE TABLE Bookings(
    BookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    CheckInDate DATETIME,
    CheckOutDate DATETIME,
    NumberOfAdults INTEGER,
    NumberOfChildren INTEGER,
    GuestID INT,
    ROOMID INT,
    FOREIGN KEY(GuestID) REFERENCES Guests(ID),
    FOREIGN KEY(ROOMID) REFERENCES Rooms(RoomID));
INSERT INTO "Bookings" VALUES(1,'2021-10-01 14:00:00','2021-10-02 11:00:00',2,0,1,1);
INSERT INTO "Bookings" VALUES(2,'2021-10-02 14:00:00','2021-10-03 11:00:00',3,1,2,2);
INSERT INTO "Bookings" VALUES(3,'2021-10-03 14:00:00','2021-10-04 11:00:00',2,1,3,3);
INSERT INTO "Bookings" VALUES(4,'2021-10-04 14:00:00','2021-10-05 11:00:00',1,0,4,4);
INSERT INTO "Bookings" VALUES(5,'2021-10-01 14:00:00','2021-10-02 11:00:00',1,1,1,1);
CREATE TABLE Guests(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    EmailAddress TEXT,
    PhoneNumber TEXT, FullName TEXT);
INSERT INTO "Guests" VALUES(1,'john.doe@example.com','123-456-7890','John Doe');
INSERT INTO "Guests" VALUES(2,'jane.smith@example.com','098-765-4321','Jane Smith');
INSERT INTO "Guests" VALUES(3,'alice.brown@example.com','555-123-4567','Alice Brown');
INSERT INTO "Guests" VALUES(4,'charlie.green@example.com','555-555-5555','Charlie Green');
INSERT INTO "Guests" VALUES(5,'david.white@example.com','555-555-5556','David White');
CREATE TABLE Rooms(
    RoomID INTEGER PRIMARY KEY AUTOINCREMENT,
    TypeOfRoom TEXT , --This ensures room type is either Guestroom or Suite
    ServiceID INT,
    FOREIGN KEY(ServiceID) REFERENCES Services(ServiceID));
INSERT INTO "Rooms" VALUES(1,'Guestroom',1);
INSERT INTO "Rooms" VALUES(2,'Guestroom',2);
INSERT INTO "Rooms" VALUES(3,'Suite',3);
INSERT INTO "Rooms" VALUES(4,'Suite',4);
CREATE TABLE Services(
    ServiceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Amenity TEXT,
    Description TEXT);
INSERT INTO "Services" VALUES(1,'Air Conditioning','Central air conditioning system.');
INSERT INTO "Services" VALUES(2,'No Air Conditioning','No air conditioning available.');
INSERT INTO "Services" VALUES(3,'Air Conditioning','Central air conditioning system.');
INSERT INTO "Services" VALUES(4,'No Air Conditioning','No air conditioning available.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('Rooms',4);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
INSERT INTO "sqlite_sequence" VALUES('Services',4);
COMMIT;
