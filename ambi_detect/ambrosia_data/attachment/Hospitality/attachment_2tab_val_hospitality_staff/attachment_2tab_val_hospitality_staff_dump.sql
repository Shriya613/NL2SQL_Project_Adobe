BEGIN TRANSACTION;
CREATE TABLE Chefs(
    chef_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    language TEXT ,
    experience_years INTEGER
);
INSERT INTO "Chefs" VALUES(1,'Juan','Spanish',10);
INSERT INTO "Chefs" VALUES(2,'Michael','English',8);
INSERT INTO "Chefs" VALUES(3,'Peter','French',6);
INSERT INTO "Chefs" VALUES(4,'Emma','German',7);
INSERT INTO "Chefs" VALUES(5,'David','Italian',9);
CREATE TABLE Customers(
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    phone_number TEXT,
    preferred_server_id INTEGER REFERENCES Servers(server_id) ON DELETE SET NULL
);
INSERT INTO "Customers" VALUES(1,'Alice','123-456-7890',1);
INSERT INTO "Customers" VALUES(2,'Bob','987-654-3210',2);
INSERT INTO "Customers" VALUES(3,'Charlie','555-555-5555',3);
INSERT INTO "Customers" VALUES(4,'Diana','111-222-3333',4);
INSERT INTO "Customers" VALUES(5,'Eva','444-444-4444',5);
CREATE TABLE Menus(
    menu_id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_id INTEGER REFERENCES Restaurants(restaurant_id) ON DELETE CASCADE,
    dish TEXT,
    price REAL
);
INSERT INTO "Menus" VALUES(1,1,'Tacos',10.99);
INSERT INTO "Menus" VALUES(2,1,'Enchiladas',12.99);
INSERT INTO "Menus" VALUES(3,2,'Coq au Vin',24.99);
INSERT INTO "Menus" VALUES(4,2,'Croissant',3.99);
INSERT INTO "Menus" VALUES(5,3,'Spaghetti Carbonara',15.99);
CREATE TABLE Orders(
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER REFERENCES Customers(customer_id) ON DELETE CASCADE,
    restaurant_id INTEGER REFERENCES Restaurants(restaurant_id) ON DELETE CASCADE,
    total_price REAL
);
INSERT INTO "Orders" VALUES(1,1,1,25.97);
INSERT INTO "Orders" VALUES(2,2,1,32.98);
INSERT INTO "Orders" VALUES(3,3,2,40.96);
INSERT INTO "Orders" VALUES(4,4,2,48.95);
INSERT INTO "Orders" VALUES(5,5,3,55.94);
CREATE TABLE Restaurants(
    restaurant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    cuisine TEXT
);
INSERT INTO "Restaurants" VALUES(1,'La Tapatia','123 Main St','Mexican');
INSERT INTO "Restaurants" VALUES(2,'Le Petit Bistro','456 Elm St','French');
INSERT INTO "Restaurants" VALUES(3,'Il Ristorante','789 Oak St','Italian');
INSERT INTO "Restaurants" VALUES(4,'The Curry House','135 Pine St','Indian');
INSERT INTO "Restaurants" VALUES(5,'El Sultan','246 Maple Ave','Middle Eastern');
CREATE TABLE Servers(
    server_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    language TEXT ,
    years_of_service INTEGER
);
INSERT INTO "Servers" VALUES(1,'Luis','Spanish',3);
INSERT INTO "Servers" VALUES(2,'Tom','English',2);
INSERT INTO "Servers" VALUES(3,'Rachel','French',1);
INSERT INTO "Servers" VALUES(4,'Lucas','German',4);
INSERT INTO "Servers" VALUES(5,'Isabella','Italian',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Chefs',5);
INSERT INTO "sqlite_sequence" VALUES('Servers',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Restaurants',5);
INSERT INTO "sqlite_sequence" VALUES('Menus',5);
INSERT INTO "sqlite_sequence" VALUES('Orders',5);
COMMIT;
