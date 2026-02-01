BEGIN TRANSACTION;
CREATE TABLE Bookings(
    BookingID INTEGER PRIMARY KEY,
    CheckInDate DATE,
    CheckOutDate DATE ,
    TotalCost REAL,
    GuestID INTEGER,
    FOREIGN KEY (GuestID) REFERENCES Guests(GuestID)
);
INSERT INTO "Bookings" VALUES(1,'2023-01-01','2023-01-05',600.0,1);
INSERT INTO "Bookings" VALUES(2,'2023-01-10','2023-01-15',1000.0,2);
INSERT INTO "Bookings" VALUES(3,'2023-01-15','2023-01-20',1500.0,3);
INSERT INTO "Bookings" VALUES(4,'2023-02-01','2023-02-05',400.0,4);
INSERT INTO "Bookings" VALUES(5,'2023-02-08','2023-02-12',750.0,5);
CREATE TABLE FamilySuites(
    SuiteID INTEGER PRIMARY KEY,
    "Room" TEXT,
    OccupancyLimit INT,
    Amenities TEXT,
    Price_per_night REAL
);
INSERT INTO "FamilySuites" VALUES(1,'A123',6,'Air conditioning, TV, WiFi, Mini fridge',150.0);
INSERT INTO "FamilySuites" VALUES(2,'B456',8,'Air conditioning, TV, WiFi, Balcony, Microwave',200.0);
INSERT INTO "FamilySuites" VALUES(3,'C789',10,'Air conditioning, TV, WiFi, Kitchenette',250.0);
INSERT INTO "FamilySuites" VALUES(4,'D012',12,'Air conditioning, TV, WiFi, Safe, Bathtub',300.0);
INSERT INTO "FamilySuites" VALUES(5,'E345',14,'Air conditioning, TV, WiFi, Kitchenette, Terrace',350.0);
CREATE TABLE Guests(
    GuestID INTEGER PRIMARY KEY,
    FullName TEXT,
    PhoneNumber TEXT,
    EmailAddress TEXT
);
INSERT INTO "Guests" VALUES(1,'John Doe','+1234567890','johndoe@example.com');
INSERT INTO "Guests" VALUES(2,'Jane Smith','+0987654321','janesmith@example.com');
INSERT INTO "Guests" VALUES(3,'Michael Johnson','+1234567890','michaeljohnson@example.com');
INSERT INTO "Guests" VALUES(4,'Emily Davis','+0987654321','emilydavis@example.com');
INSERT INTO "Guests" VALUES(5,'David Brown','+1234567890','davidbrown@example.com');
CREATE TABLE Payments(
    PaymentID INTEGER PRIMARY KEY AUTOINCREMENT,
    Amount PRECISION,
    DatePaid DATE,
    CardType TEXT,
    BookingID INTEGER,
    FOREIGN KEY (BookingID) REFERENCES Bookings(BookingID)
);
INSERT INTO "Payments" VALUES(1,600,'2023-01-02','Visa',1);
INSERT INTO "Payments" VALUES(2,1000,'2023-01-07','Mastercard',2);
INSERT INTO "Payments" VALUES(3,1500,'2023-01-14','American Express',3);
INSERT INTO "Payments" VALUES(4,400,'2023-02-01','Visa',4);
INSERT INTO "Payments" VALUES(5,750,'2023-02-08','Mastercard',5);
CREATE TABLE Reviews(
    ReviewID INTEGER PRIMARY KEY,
    Rating INT ,
    Comment TEXT,
    SubmissionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    GuestID INTEGER,
    FOREIGN KEY (GuestID) REFERENCES Guests(GuestID)
);
INSERT INTO "Reviews" VALUES(1,5,'Great stay! Excellent service.','2023-01-06',1);
INSERT INTO "Reviews" VALUES(2,4,'Very comfortable room, good location.','2023-01-12',2);
INSERT INTO "Reviews" VALUES(3,3,'The suite was nice but a bit expensive.','2023-01-18',3);
INSERT INTO "Reviews" VALUES(4,5,'Amazing experience! Highly recommended.','2023-02-05',4);
INSERT INTO "Reviews" VALUES(5,4,'Good value for money, would return.','2023-02-15',5);
CREATE TABLE "SingleRooms"(
    RoomID INTEGER PRIMARY KEY, "Room" TEXT,
    NumberOfBeds INT,
    View TEXT,
    Services TEXT,
    Price_per_night REAL
);
INSERT INTO "SingleRooms" VALUES(1,'101',1,'City view','TV, WiFi, Air conditioning',150.0);
INSERT INTO "SingleRooms" VALUES(2,'102',1,'Mountain view','TV, WiFi, Minibar',100.0);
INSERT INTO "SingleRooms" VALUES(3,'103',2,'Garden view','TV, WiFi, Air conditioning, Coffee/Tea maker',120.0);
INSERT INTO "SingleRooms" VALUES(4,'104',2,'Pool view','TV, WiFi, Hairdryer, Ironing facilities',130.0);
INSERT INTO "SingleRooms" VALUES(5,'105',1,'Sea view','TV, WiFi, Air conditioning, Refrigerator',180.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Payments',5);
COMMIT;
