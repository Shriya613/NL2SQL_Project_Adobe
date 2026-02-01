BEGIN TRANSACTION;
CREATE TABLE Beverages (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Type TEXT ,
    Price REAL
);
INSERT INTO "Beverages" VALUES(1,'Coca Cola','Soft Drink',2.99);
INSERT INTO "Beverages" VALUES(2,'Red Wine','Wine',8.99);
INSERT INTO "Beverages" VALUES(3,'Whiskey Sour','Cocktail',7.99);
INSERT INTO "Beverages" VALUES(4,'Beer','Alcoholic Beverage',5.99);
INSERT INTO "Beverages" VALUES(5,'Champagne','Sparkling Wine',15.99);
INSERT INTO "Beverages" VALUES(6,'Pinot Noir','Wine',10.99);
INSERT INTO "Beverages" VALUES(7,'Old Fashioned','Cocktail',8.99);
CREATE TABLE Employees (
    id INTEGER PRIMARY KEY,
    Professional_id INTEGER,
    Date_of_birth DATE,
    Email TEXT,
    Phone_number VARCHAR(15),
    FOREIGN KEY (Professional_id) REFERENCES Professionals(id)
);
INSERT INTO "Employees" VALUES(1,1,'2000-01-01','john.doe@example.com','+1234567890');
INSERT INTO "Employees" VALUES(2,2,'2001-02-02','jane.smith@example.com','+0987654321');
INSERT INTO "Employees" VALUES(3,3,'1999-03-03','alice.williams@example.com','+1112223334');
INSERT INTO "Employees" VALUES(4,4,'1998-04-04','bob.johnson@example.com','+44332211556');
INSERT INTO "Employees" VALUES(5,5,'1997-05-05','alex.martinez@example.com','+7788990011');
INSERT INTO "Employees" VALUES(6,6,'1996-06-06','alex.turner@example.com','+2233445566');
CREATE TABLE Orders (
    id INTEGER PRIMARY KEY,
    Ordered_by INTEGER,
    Item_name TEXT,
    Quantity INTEGER,
    Date_ordered TIMESTAMP,
    Total_amount REAL,
    FOREIGN KEY (Ordered_by) REFERENCES Reservations(id)
);
INSERT INTO "Orders" VALUES(1,1,'Steak',2,'2023-01-01 19:30:00',70.0);
INSERT INTO "Orders" VALUES(2,2,'Pizza Margherita',3,'2023-01-02 20:30:00',45.0);
INSERT INTO "Orders" VALUES(3,3,'White Burgundy',4,'2023-01-03 19:30:00',53.96);
INSERT INTO "Orders" VALUES(4,4,'Bloody Mary',2,'2023-01-04 20:30:00',17.98);
CREATE TABLE Professionals (
    id INTEGER PRIMARY KEY,
    Position TEXT 
, FullName TEXT);
INSERT INTO "Professionals" VALUES(1,'Manager','John Doe');
INSERT INTO "Professionals" VALUES(2,'Chef','Jane Smith');
INSERT INTO "Professionals" VALUES(3,'Sommelier','Alice Williams');
INSERT INTO "Professionals" VALUES(4,'Bartender','Bob Johnson');
INSERT INTO "Professionals" VALUES(5,'Sommelier','Alex Martinez');
INSERT INTO "Professionals" VALUES(6,'Bartender','Alex Turner');
CREATE TABLE Reservations (
    id INTEGER PRIMARY KEY,
    Customer_id INTEGER,
    Table_Number INTEGER,
    Reservation_date TIMESTAMP,
    Party_size INTEGER,
    Total_cost REAL,
    Foreign Key (Customer_id) References Customers(id)
);
INSERT INTO "Reservations" VALUES(1,1,1,'2023-01-01 19:00:00',4,200.0);
INSERT INTO "Reservations" VALUES(2,2,2,'2023-01-02 20:00:00',2,100.0);
INSERT INTO "Reservations" VALUES(3,3,3,'2023-01-03 19:00:00',6,300.0);
INSERT INTO "Reservations" VALUES(4,4,4,'2023-01-04 20:00:00',5,250.0);
CREATE TABLE Restaurants (
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Opening_hours TEXT
);
INSERT INTO "Restaurants" VALUES(1,'The Fancy Bistro','New York City','Monday - Sunday: 10 AM - 11 PM');
INSERT INTO "Restaurants" VALUES(2,'Le Jardin Secret','Paris','Monday - Friday: 10 AM - 10 PM, Saturday - Sunday: 11 AM - 11 PM');
COMMIT;
