BEGIN TRANSACTION;
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT,
    PhoneNumber VARCHAR(15)
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com','+123-456-7890');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com','+098-765-4321');
INSERT INTO "Customers" VALUES(3,'Alice','Brown','alice.brown@example.com','+112-345-6789');
INSERT INTO "Customers" VALUES(4,'Mike','Johnson','mike.johnson@example.com','+234-567-8901');
INSERT INTO "Customers" VALUES(5,'Emily','White','emily.white@example.com','+356-789-0123');
CREATE TABLE Hotels (
    HotelID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    City TEXT,
    State TEXT,
    PostalCode TEXT,
    Country TEXT
);
INSERT INTO "Hotels" VALUES(1,'Grand Central','1 Grand Central Plaza','New York','NY','10017','USA');
INSERT INTO "Hotels" VALUES(2,'The Ritz Carlton','2 Ritz Carlton Drive','Los Angeles','CA','90005','USA');
INSERT INTO "Hotels" VALUES(3,'Hilton Garden Inn','3 Hilton Garden Inn Boulevard','Chicago','IL','60601','USA');
INSERT INTO "Hotels" VALUES(4,'Fairmont San Francisco','4 Nob Hill Ave','San Francisco','CA','94108','USA');
INSERT INTO "Hotels" VALUES(5,'Marriott Marquis Houston','5 Marriott Marquis Blvd','Houston','TX','77002','USA');
CREATE TABLE Reservations (
    ReservationID INTEGER PRIMARY KEY,
    CheckInDate DATE,
    CheckOutDate DATE,
    GuestID INTEGER REFERENCES Customers(CustomerID),
    TotalCost DECIMAL(8,2),
    PaymentMethod TEXT
);
INSERT INTO "Reservations" VALUES(1,'2021-08-01','2021-08-05',1,500,'Credit Card');
INSERT INTO "Reservations" VALUES(2,'2021-08-06','2021-08-10',2,800,'PayPal');
INSERT INTO "Reservations" VALUES(3,'2021-08-11','2021-08-15',3,600,'Cash');
INSERT INTO "Reservations" VALUES(4,'2021-08-16','2021-08-20',4,700,'Credit Card');
INSERT INTO "Reservations" VALUES(5,'2021-08-21','2021-08-25',5,900,'Debit Card');
CREATE TABLE "Reviews" (
    ReviewID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER REFERENCES Customers(CustomerID),
    HotelID INTEGER REFERENCES Hotels(HotelID),
    Rating TEXT,
    Date_Of_Review DATE DEFAULT CURRENT_DATE
);
INSERT INTO "Reviews" VALUES(1,1,1,'Great experience! The staff was friendly and helpful.','2021-08-03');
INSERT INTO "Reviews" VALUES(2,2,2,'Excellent location and comfortable rooms.','2021-08-05');
INSERT INTO "Reviews" VALUES(3,3,3,'Nice hotel but could use some updates.','2021-08-09');
INSERT INTO "Reviews" VALUES(4,4,4,'Awesome service and great amenities.','2021-08-12');
INSERT INTO "Reviews" VALUES(5,5,5,'I enjoyed my stay at this lovely hotel.','2021-08-15');
CREATE TABLE Rooms (
    RoomID INTEGER PRIMARY KEY,
    HotelID INTEGER REFERENCES Hotels(HotelID),
    Number INT,
    Type TEXT,
    Amenities TEXT
);
INSERT INTO "Rooms" VALUES(1,1,101,'Deluxe King','Free WiFi, Air conditioning, Flat-screen TV');
INSERT INTO "Rooms" VALUES(2,1,102,'Standard Queen','Free WiFi, Air conditioning, Flat-screen TV');
INSERT INTO "Rooms" VALUES(3,2,201,'Executive Suite','Free WiFi, Air conditioning, Flat-screen TV, Kitchenette');
INSERT INTO "Rooms" VALUES(4,2,202,'Premium Twin','Free WiFi, Air conditioning, Flat-screen TV');
INSERT INTO "Rooms" VALUES(5,3,301,'Family Suite','Free WiFi, Air conditioning, Flat-screen TV, Separate living area');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Reviews',5);
COMMIT;
