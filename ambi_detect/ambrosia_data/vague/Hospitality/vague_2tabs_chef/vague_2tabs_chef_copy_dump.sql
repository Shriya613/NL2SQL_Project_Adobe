BEGIN TRANSACTION;
CREATE TABLE "Chef" (
    chef_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Chef" VALUES(1,'John','Doe');
INSERT INTO "Chef" VALUES(2,'Jane','Smith');
INSERT INTO "Chef" VALUES(3,'Michael','Johnson');
CREATE TABLE Menu_items (
    menu_item_id INTEGER PRIMARY KEY,
    item_name TEXT,
    description TEXT,
    price DECIMAL(5,2) 
);
INSERT INTO "Menu_items" VALUES(1,'Steak','Grilled Steak with Veggies',30.99);
INSERT INTO "Menu_items" VALUES(2,'Salmon','Grilled Salmon with Rice',25.8);
INSERT INTO "Menu_items" VALUES(3,'Pasta','Spaghetti Bolognese',15.6);
INSERT INTO "Menu_items" VALUES(4,'Pizza','Margherita Pizza',12.5);
INSERT INTO "Menu_items" VALUES(5,'Burger','Cheeseburger and Fries',10.8);
CREATE TABLE Order_details (
    detail_id INTEGER PRIMARY KEY,
    quantity INTEGER ,
    menu_item_id INTEGER,
    order_id INTEGER,
    FOREIGN KEY(menu_item_id) REFERENCES Menu_items(menu_item_id),
    FOREIGN KEY(order_id) REFERENCES Orders(order_id)
);
INSERT INTO "Order_details" VALUES(1,2,1,1);
INSERT INTO "Order_details" VALUES(2,1,2,2);
INSERT INTO "Order_details" VALUES(3,1,3,3);
INSERT INTO "Order_details" VALUES(4,1,1,4);
INSERT INTO "Order_details" VALUES(5,1,4,5);
CREATE TABLE Orders (
    order_id INTEGER PRIMARY KEY,
    customer_name TEXT,
    date DATE,
    total_amount DECIMAL(7,2),
    restaurant_id INTEGER,
    FOREIGN KEY(restaurant_id) REFERENCES Restaurant(restaurant_id)
);
INSERT INTO "Orders" VALUES(1,'Customer A','2021-08-10',60.2,1);
INSERT INTO "Orders" VALUES(2,'Customer B','2021-08-11',45.5,2);
INSERT INTO "Orders" VALUES(3,'Customer C','2021-08-12',75.3,3);
INSERT INTO "Orders" VALUES(4,'Customer D','2021-08-13',80.5,1);
INSERT INTO "Orders" VALUES(5,'Customer E','2021-08-14',55.2,2);
CREATE TABLE "Restaurant" (
    restaurant_id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT,
    executive_chef_id INTEGER,
    FOREIGN KEY(executive_chef_id) REFERENCES "Chef"(chef_id));
INSERT INTO "Restaurant" VALUES(1,'The Ritz','Main Street',1);
INSERT INTO "Restaurant" VALUES(2,'Le Grande','East Side',2);
INSERT INTO "Restaurant" VALUES(3,'Ocean View','Beach Road',3);
COMMIT;
