BEGIN TRANSACTION;
CREATE TABLE Bookings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    EventSpaceId INTEGER,
    Date DATE,
    Time TIME,
    NumberOfGuests INT,
    FOREIGN KEY(EventSpaceId) REFERENCES EventSpaces(id));
INSERT INTO "Bookings" VALUES(1,1,'2023-01-01','15:00',100);
INSERT INTO "Bookings" VALUES(2,2,'2023-01-02','19:00',80);
INSERT INTO "Bookings" VALUES(3,3,'2023-01-03','10:00',120);
INSERT INTO "Bookings" VALUES(4,4,'2023-01-04','14:00',60);
INSERT INTO "Bookings" VALUES(5,5,'2023-01-05','18:00',150);
INSERT INTO "Bookings" VALUES(6,6,'2023-01-06','21:00',100);
INSERT INTO "Bookings" VALUES(7,7,'2023-01-07','15:00',90);
INSERT INTO "Bookings" VALUES(8,8,'2023-01-08','19:00',70);
INSERT INTO "Bookings" VALUES(9,9,'2023-01-09','10:00',80);
INSERT INTO "Bookings" VALUES(10,10,'2023-01-10','14:00',110);
INSERT INTO "Bookings" VALUES(11,11,'2023-01-11','18:00',120);
INSERT INTO "Bookings" VALUES(12,12,'2023-01-12','21:00',130);
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber TEXT);
INSERT INTO "Customers" VALUES(1,'John','Doe','johndoe@example.com','+1-555-555-5555');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','janesmith@example.com','+1-555-555-5556');
INSERT INTO "Customers" VALUES(3,'Michael','Williams','mwilliams@example.com','+1-555-555-5557');
INSERT INTO "Customers" VALUES(4,'Emily','Johnson','emilyjohnson@example.com','+1-555-555-5558');
INSERT INTO "Customers" VALUES(5,'David','Brown','davidbrown@example.com','+1-555-555-5559');
INSERT INTO "Customers" VALUES(6,'Laura','Taylor','lataylor@example.com','+1-555-555-5560');
CREATE TABLE EventSpaces(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Event_Space TEXT ,
    Capacity INT,
    Address TEXT,
    ContactInfo TEXT);
INSERT INTO "EventSpaces" VALUES(1,'Grand Ballroom','Banquet Hall',200,'123 Main St','+1-800-123-4567');
INSERT INTO "EventSpaces" VALUES(2,'Presidents Suite','Banquet Hall',150,'456 Elm Street','+1-800-987-6543');
INSERT INTO "EventSpaces" VALUES(3,'Meeting Room A','Conference Room',200,'789 Oak Avenue','+1-800-555-1234');
INSERT INTO "EventSpaces" VALUES(4,'Boardroom B','Conference Room',100,'135 Pine Drive','+1-800-555-5678');
INSERT INTO "EventSpaces" VALUES(5,'Garden Terrace','Outdoor Space',120,'404 Maple Road','+1-800-111-2222');
INSERT INTO "EventSpaces" VALUES(6,'Rooftop Lounge','Indoor/Outdoor Space',80,'567 Cedar Lane','+1-800-333-4444');
INSERT INTO "EventSpaces" VALUES(7,'Wintergarten','Banquet Hall',250,'987 Palm Boulevard','+1-800-777-8888');
INSERT INTO "EventSpaces" VALUES(8,'The Atrium','Banquet Hall',180,'321 Birch Way','+1-800-900-1000');
INSERT INTO "EventSpaces" VALUES(9,'Executive Boardroom','Conference Room',120,'777 Willow Circle','+1-800-111-2222');
INSERT INTO "EventSpaces" VALUES(10,'Pavilion','Outdoor Space',150,'123 Spruce Trail','+1-800-333-4444');
CREATE TABLE Payments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    BookingId INTEGER,
    Amount DECIMAL(10, 2),
    PaymentDate DATE,
    CustomerId INTEGER,
    FOREIGN KEY(BookingId) REFERENCES Bookings(id),
    FOREIGN KEY(CustomerId) REFERENCES Customers(id));
INSERT INTO "Payments" VALUES(1,1,1000,'2023-01-01',1);
INSERT INTO "Payments" VALUES(2,2,800,'2023-01-02',2);
INSERT INTO "Payments" VALUES(3,3,1200,'2023-01-03',3);
INSERT INTO "Payments" VALUES(4,4,900,'2023-01-04',4);
INSERT INTO "Payments" VALUES(5,5,1100,'2023-01-05',5);
INSERT INTO "Payments" VALUES(6,6,1000,'2023-01-06',1);
INSERT INTO "Payments" VALUES(7,7,900,'2023-01-07',2);
INSERT INTO "Payments" VALUES(8,8,1200,'2023-01-08',3);
INSERT INTO "Payments" VALUES(9,9,1100,'2023-01-09',4);
INSERT INTO "Payments" VALUES(10,10,1000,'2023-01-10',5);
INSERT INTO "Payments" VALUES(11,11,1200,'2023-01-11',6);
INSERT INTO "Payments" VALUES(12,12,1300,'2023-01-12',7);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('EventSpaces',10);
INSERT INTO "sqlite_sequence" VALUES('Bookings',12);
INSERT INTO "sqlite_sequence" VALUES('Customers',6);
INSERT INTO "sqlite_sequence" VALUES('Payments',12);
COMMIT;
