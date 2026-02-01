BEGIN TRANSACTION;
CREATE TABLE Bookings (
    BookingID INTEGER PRIMARY KEY AUTOINCREMENT,
    GuestID INTEGER,
    RoomID INTEGER,
    CheckInDate DATE,
    CheckOutDate DATE,
    TotalCost REAL,
    FOREIGN KEY(GuestID) REFERENCES Guests(GuestID),
    FOREIGN KEY(RoomID) REFERENCES Rooms(RoomID)
);
INSERT INTO "Bookings" VALUES(1,1,1,'2022-01-01','2022-01-05',600.0);
INSERT INTO "Bookings" VALUES(2,2,2,'2022-01-10','2022-01-14',800.0);
INSERT INTO "Bookings" VALUES(3,3,3,'2022-02-01','2022-02-05',720.0);
INSERT INTO "Bookings" VALUES(4,4,4,'2022-03-01','2022-03-05',960.0);
INSERT INTO "Bookings" VALUES(5,5,5,'2022-04-01','2022-04-05',840.0);
CREATE TABLE Guests (
    GuestID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber VARCHAR(15) UNIQUE
);
INSERT INTO "Guests" VALUES(1,'John','Doe','john.doe@example.com','+1234567890');
INSERT INTO "Guests" VALUES(2,'Jane','Smith','jane.smith@example.com','+0987654321');
INSERT INTO "Guests" VALUES(3,'Alice','Brown','alice.brown@example.com','+1122334455');
INSERT INTO "Guests" VALUES(4,'Bob','White','bob.white@example.com','+2233445566');
INSERT INTO "Guests" VALUES(5,'Charlie','Black','charlie.black@example.com','+3344556677');
CREATE TABLE "Hotels" (
    HotelID INTEGER PRIMARY KEY AUTOINCREMENT,
    Opening_Date INTEGER,
    Name TEXT,
    Address TEXT,
    City TEXT,
    Country TEXT
);
INSERT INTO "Hotels" VALUES(1,2000,'Grand Plaza','123 Main St','New York','USA');
INSERT INTO "Hotels" VALUES(2,2018,'The Ritz','456 Broadway','London','UK');
INSERT INTO "Hotels" VALUES(3,2005,'Ocean View','789 Beach Ave','Miami','USA');
CREATE TABLE Rooms (
    RoomID INTEGER PRIMARY KEY AUTOINCREMENT,
    HotelID INTEGER,
    Type TEXT,
    Capacity INTEGER,
    Rate REAL,
    FOREIGN KEY(HotelID) REFERENCES Hotels(HotelID)
);
INSERT INTO "Rooms" VALUES(1,1,'Standard',4,150.0);
INSERT INTO "Rooms" VALUES(2,1,'Deluxe',2,200.0);
INSERT INTO "Rooms" VALUES(3,2,'Luxury',1,300.0);
INSERT INTO "Rooms" VALUES(4,2,'Suite',2,400.0);
INSERT INTO "Rooms" VALUES(5,3,'Sea View',4,180.0);
CREATE TABLE Services (
    ServiceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    Cost REAL,
    Available BOOLEAN DEFAULT 0
);
INSERT INTO "Services" VALUES(1,'Laundry service',5.0,1);
INSERT INTO "Services" VALUES(2,'Room cleaning',10.0,1);
INSERT INTO "Services" VALUES(3,'Shuttle to airport',50.0,1);
INSERT INTO "Services" VALUES(4,'Breakfast buffet',20.0,1);
INSERT INTO "Services" VALUES(5,'Gym access',25.0,1);
CREATE TABLE StaffMembers (
    StaffMemberID INTEGER PRIMARY KEY AUTOINCREMENT,
    FullName TEXT,
    Position TEXT,
    Department TEXT,
    HireDate DATE
);
INSERT INTO "StaffMembers" VALUES(1,'Emma Jones','Receptionist','Front Desk','2021-05-15');
INSERT INTO "StaffMembers" VALUES(2,'Tom Smith','Housekeeping Supervisor','Housekeeping','2021-06-01');
INSERT INTO "StaffMembers" VALUES(3,'Ann Brown','Food and Beverage Manager','Restaurant','2021-07-15');
INSERT INTO "StaffMembers" VALUES(4,'Jack White','Maintenance Supervisor','Engineering','2021-08-01');
INSERT INTO "StaffMembers" VALUES(5,'Peter Black','Concierge','General Services','2021-09-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Rooms',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('StaffMembers',5);
INSERT INTO "sqlite_sequence" VALUES('Services',5);
INSERT INTO "sqlite_sequence" VALUES('Hotels',3);
COMMIT;
