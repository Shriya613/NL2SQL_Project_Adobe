BEGIN TRANSACTION;
CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    address TEXT,
    phone_number TEXT
);
INSERT INTO "Customers" VALUES(1,'John Doe','john@example.com','123 Main St, Anytown, USA','+1-123-456-7890');
INSERT INTO "Customers" VALUES(2,'Jane Smith','jane@example.com','456 Elm St, Anycity, USA','+1-987-654-3210');
CREATE TABLE Employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    position TEXT,
    hire_date DATE,
    wage DECIMAL(8,2) 
);
INSERT INTO "Employees" VALUES(1,'Michael','Smith','Manager','2022-05-01',50000);
INSERT INTO "Employees" VALUES(2,'Emily','Johnson','Server','2022-06-01',25000);
CREATE TABLE MenuItems (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_id INTEGER,
    item_name TEXT,
    description TEXT,
    price DECIMAL(10,2) ,
    FOREIGN KEY (restaurant_id) REFERENCES Restaurants (restaurant_id) ON DELETE CASCADE
);
INSERT INTO "MenuItems" VALUES(1,1,'Steak','Grilled steak with garlic butter',29.99);
INSERT INTO "MenuItems" VALUES(2,1,'Seafood Platter','Assorted seafood with lemon herb sauce',34.99);
CREATE TABLE Reservations (
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT,
    phone_number TEXT,
    date DATE,
    time TIME
);
INSERT INTO "Reservations" VALUES(1,'John Doe','+1-123-456-7890','2023-01-01','18:30');
INSERT INTO "Reservations" VALUES(2,'Jane Smith','+1-987-654-3210','2023-01-02','19:00');
CREATE TABLE "Restaurants" (
    restaurant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    Chef TEXT,
    cuisine TYPE
);
INSERT INTO "Restaurants" VALUES(1,'Gourmet Grill','Downtown','Chef Johnson','Steaks & Seafood');
CREATE TABLE Reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    rating INTEGER ,
    comments TEXT,
    FOREIGN KEY (customer_id) REFERENCES Customers (customer_id) ON DELETE CASCADE
);
INSERT INTO "Reviews" VALUES(1,1,5,'Great food and service!');
INSERT INTO "Reviews" VALUES(2,2,4,'Nice place but could use improvement on presentation.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Reservations',2);
INSERT INTO "sqlite_sequence" VALUES('Customers',2);
INSERT INTO "sqlite_sequence" VALUES('MenuItems',2);
INSERT INTO "sqlite_sequence" VALUES('Reviews',2);
INSERT INTO "sqlite_sequence" VALUES('Employees',2);
INSERT INTO "sqlite_sequence" VALUES('Restaurants',1);
COMMIT;
