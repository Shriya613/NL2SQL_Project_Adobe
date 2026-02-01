BEGIN TRANSACTION;
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phone TEXT,
    preferred_language TEXT
);
CREATE TABLE MenuItems(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_id INTEGER,
    item_name TEXT,
    price REAL,
    FOREIGN KEY(restaurant_id) REFERENCES Restaurant(id)
);
CREATE TABLE Orders(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_price REAL,
    paid BOOLEAN,
    FOREIGN KEY(customer_id) REFERENCES Customers(id)
);
CREATE TABLE Restaurant(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    Service_format TEXT ,
    Hours TEXT
);
INSERT INTO "Restaurant" VALUES(1,'The Royal Buffet','123 Main St','Buffet','7.30 a.m. - 4 p.m.');
INSERT INTO "Restaurant" VALUES(2,'Fresh Foods Cafe','456 Market St','Buffet','8 a.m. - 9 p.m.');
INSERT INTO "Restaurant" VALUES(3,'Java Junction','789 Elm St','Coffee Shop','7.30 a.m. - 4 p.m.');
INSERT INTO "Restaurant" VALUES(4,'Deli Delights','135 Oak St','Coffee Shop','6 a.m. - 6 p.m.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Restaurant',4);
COMMIT;
