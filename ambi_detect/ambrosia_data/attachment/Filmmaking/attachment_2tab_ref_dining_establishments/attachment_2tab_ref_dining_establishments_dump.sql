BEGIN TRANSACTION;
CREATE TABLE Cafe(
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    specialty TEXT
);
INSERT INTO "Cafe" VALUES(6,'The Herb House','Downtown','Herbal Teas');
INSERT INTO "Cafe" VALUES(7,'Garden Delights','Uptown','Organic Coffee');
INSERT INTO "Cafe" VALUES(8,'Tea Time','Northside','Afternoon Tea');
INSERT INTO "Cafe" VALUES(9,'Vegan Treats','Eastside','Vegetarian Options');
INSERT INTO "Cafe" VALUES(10,'Brew Haven','Westside','Specialty Roasts');
CREATE TABLE Cafe_Menus(
    cafe_id INTEGER,
    menu_option_id INTEGER,
    FOREIGN KEY(cafe_id) REFERENCES Cafe(id),
    FOREIGN KEY(menu_option_id) REFERENCES Menu_Options(id),
    PRIMARY KEY(cafe_id, menu_option_id)
);
INSERT INTO "Cafe_Menus" VALUES(6,10);
INSERT INTO "Cafe_Menus" VALUES(7,1);
INSERT INTO "Cafe_Menus" VALUES(8,2);
INSERT INTO "Cafe_Menus" VALUES(9,3);
INSERT INTO "Cafe_Menus" VALUES(10,4);
CREATE TABLE Menu_Options(
    id INTEGER PRIMARY KEY,
    option_name TEXT UNIQUE
);
INSERT INTO "Menu_Options" VALUES(1,'Spaghetti Carbonara');
INSERT INTO "Menu_Options" VALUES(2,'Margherita Pizza');
INSERT INTO "Menu_Options" VALUES(3,'Caprese Salad');
INSERT INTO "Menu_Options" VALUES(4,'Lasagna');
INSERT INTO "Menu_Options" VALUES(5,'Tiramisu');
INSERT INTO "Menu_Options" VALUES(6,'Ratatouille');
INSERT INTO "Menu_Options" VALUES(7,'Quiche Lorraine');
INSERT INTO "Menu_Options" VALUES(8,'Grilled Vegetable Panini');
INSERT INTO "Menu_Options" VALUES(9,'Minestrone Soup');
INSERT INTO "Menu_Options" VALUES(10,'Gelato');
CREATE TABLE Restaurant_Menus(
    restaurant_id INTEGER,
    menu_option_id INTEGER,
    FOREIGN KEY(restaurant_id) REFERENCES Restaurants(id),
    FOREIGN KEY(menu_option_id) REFERENCES Menu_Options(id),
    PRIMARY KEY(restaurant_id, menu_option_id)
);
INSERT INTO "Restaurant_Menus" VALUES(1,1);
INSERT INTO "Restaurant_Menus" VALUES(1,2);
INSERT INTO "Restaurant_Menus" VALUES(2,3);
INSERT INTO "Restaurant_Menus" VALUES(2,4);
INSERT INTO "Restaurant_Menus" VALUES(3,5);
INSERT INTO "Restaurant_Menus" VALUES(3,6);
INSERT INTO "Restaurant_Menus" VALUES(4,7);
INSERT INTO "Restaurant_Menus" VALUES(4,8);
INSERT INTO "Restaurant_Menus" VALUES(5,9);
CREATE TABLE Restaurants(
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    cuisine TEXT
);
INSERT INTO "Restaurants" VALUES(1,'The Garden Bistro','Downtown','Italian');
INSERT INTO "Restaurants" VALUES(2,'Le Jardin','Uptown','French');
INSERT INTO "Restaurants" VALUES(3,'Green Oasis','Northside','Mediterranean');
INSERT INTO "Restaurants" VALUES(4,'La Veggie Cucina','Eastside','Vegetarian');
INSERT INTO "Restaurants" VALUES(5,'Pasta Palace','Westside','Italian');
COMMIT;
