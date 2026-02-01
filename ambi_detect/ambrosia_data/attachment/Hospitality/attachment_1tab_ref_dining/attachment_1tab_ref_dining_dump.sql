BEGIN TRANSACTION;
CREATE TABLE Customers(
    CustomerID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT,
    PhoneNumber TEXT
);
INSERT INTO "Customers" VALUES(1,'John','Smith','johnsmith@example.com','+123456789');
INSERT INTO "Customers" VALUES(2,'Jane','Doe','janedoe@example.com','+098765432');
CREATE TABLE FoodTypes(
    FoodTypeID INTEGER PRIMARY KEY AUTOINCREMENT,
    TypeName TEXT
);
INSERT INTO "FoodTypes" VALUES(1,'Prepared Meal');
INSERT INTO "FoodTypes" VALUES(2,'Takeaway');
CREATE TABLE Orders(
    OrderID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    TotalAmount NUMERIC,
    OrderDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY(OrderID) REFERENCES Products(ProductID)
);
INSERT INTO "Orders" VALUES(1,1,100,'2024-03-13 11:03:54');
INSERT INTO "Orders" VALUES(2,2,150,'2024-03-13 11:03:54');
CREATE TABLE Products(
    ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Dining INT DEFAULT 0 , -- 0=None, 1=Meals Prepared On Site, 2=Takeaway, 3=Other
    Best_Before TEXT,
    FOREIGN KEY(Dining) REFERENCES FoodTypes(FoodTypeID)
);
INSERT INTO "Products" VALUES(1,'Pizza',1,'10.11.2018');
INSERT INTO "Products" VALUES(2,'Burger',1,'11.11.2018');
INSERT INTO "Products" VALUES(3,'Salad',2,'10.11.2018');
INSERT INTO "Products" VALUES(4,'Coffee',2,'12.11.2018');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customers',2);
INSERT INTO "sqlite_sequence" VALUES('FoodTypes',2);
INSERT INTO "sqlite_sequence" VALUES('Products',4);
INSERT INTO "sqlite_sequence" VALUES('Orders',2);
COMMIT;
