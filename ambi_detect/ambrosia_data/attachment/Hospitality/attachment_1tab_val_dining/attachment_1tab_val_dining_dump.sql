BEGIN TRANSACTION;
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber TEXT
);
INSERT INTO "Customers" VALUES(1,'John','Doe','john.doe@example.com','+123456789');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','jane.smith@example.com','+098765432');
INSERT INTO "Customers" VALUES(3,'Alice','Brown','alice.brown@example.com','+1123456789');
INSERT INTO "Customers" VALUES(4,'Charlie','White','charlie.white@example.com','+0098765432');
INSERT INTO "Customers" VALUES(5,'David','Black','david.black@example.com','+11123456789');
CREATE TABLE Meals (
    MealID INTEGER PRIMARY KEY AUTOINCREMENT,
    MealType TEXT ,
    BestBefore DATE,
    Description TEXT,
    Price DECIMAL(7, 2) -- Assuming maximum price is 999999.99
);
INSERT INTO "Meals" VALUES(1,'Prepared Meal','10.11.2018','A delicious steak dinner.',29.99);
INSERT INTO "Meals" VALUES(2,'Prepared Meal','11.12.2018','Grilled salmon with asparagus.',19.99);
INSERT INTO "Meals" VALUES(3,'Takeaway','10.11.2018','Fried chicken combo meal.',9.99);
INSERT INTO "Meals" VALUES(4,'Takeaway','11.12.2018','Veggie burger and fries.',8.99);
INSERT INTO "Meals" VALUES(5,'Breakfast','12.13.2018','Eggs Benedict with hollandaise sauce.',14.99);
CREATE TABLE Orders (
    OrderID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    MealID INTEGER,
    Quantity INTEGER,
    OrderDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (MealID) REFERENCES Meals(MealID)
);
INSERT INTO "Orders" VALUES(1,1,1,2,'2024-03-07 21:09:43');
INSERT INTO "Orders" VALUES(2,2,2,1,'2024-03-07 21:09:43');
INSERT INTO "Orders" VALUES(3,3,3,3,'2024-03-07 21:09:43');
INSERT INTO "Orders" VALUES(4,4,4,1,'2024-03-07 21:09:43');
INSERT INTO "Orders" VALUES(5,5,5,2,'2024-03-07 21:09:43');
CREATE TABLE Restaurants (
    RestaurantID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    OpeningTime TEXT,
    ClosingTime TEXT
);
INSERT INTO "Restaurants" VALUES(1,'The Gourmet Place','123 Main St','08:00:00','22:00:00');
INSERT INTO "Restaurants" VALUES(2,'Fast Bites','456 Market St','07:00:00','23:00:00');
INSERT INTO "Restaurants" VALUES(3,'Cozy Cafe','789 High St','09:00:00','21:00:00');
INSERT INTO "Restaurants" VALUES(4,'Deli Delight','321 Ocean Blvd','10:00:00','20:00:00');
INSERT INTO "Restaurants" VALUES(5,'Pizza Palace','987 Second Ave','11:00:00','23:59:59');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Restaurants',5);
INSERT INTO "sqlite_sequence" VALUES('Meals',5);
INSERT INTO "sqlite_sequence" VALUES('Orders',5);
COMMIT;
