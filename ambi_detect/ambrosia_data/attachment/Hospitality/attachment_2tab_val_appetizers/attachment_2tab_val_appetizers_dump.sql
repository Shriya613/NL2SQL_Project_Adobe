BEGIN TRANSACTION;
CREATE TABLE Beverages(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Price REAL ,
    Quantity INTEGER ,
    BuffetName TEXT
);
INSERT INTO "Beverages" VALUES(1,'Cola',1.99,50,'Hilton');
INSERT INTO "Beverages" VALUES(2,'Espresso',3.99,40,'Marriott');
INSERT INTO "Beverages" VALUES(3,'Orange Juice',2.99,35,'Hyatt');
INSERT INTO "Beverages" VALUES(4,'Sparkling Water',1.99,40,'Wyndham');
INSERT INTO "Beverages" VALUES(5,'Tea',2.99,30,'Intercontinental');
CREATE TABLE Customers(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    ContactNumber TEXT,
    Email TEXT UNIQUE
);
INSERT INTO "Customers" VALUES(1,'Samuel','Anderson','+1234567890','sam@example.com');
INSERT INTO "Customers" VALUES(2,'David','Miller','+1234567891','david@example.com');
INSERT INTO "Customers" VALUES(3,'Rachel','Thomas','+1234567892','rachel@example.com');
INSERT INTO "Customers" VALUES(4,'James','Jackson','+1234567893','james@example.com');
INSERT INTO "Customers" VALUES(5,'Laura','White','+1234567894','laura@example.com');
CREATE TABLE Desserts(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Price REAL ,
    Quantity INTEGER ,
    BuffetName TEXT
);
INSERT INTO "Desserts" VALUES(1,'Cheesecake',4.99,25,'Hilton');
INSERT INTO "Desserts" VALUES(2,'Tiramisu',5.99,30,'Marriott');
INSERT INTO "Desserts" VALUES(3,'Banana Split',6.99,20,'Hyatt');
INSERT INTO "Desserts" VALUES(4,'Apple Pie',5.99,15,'Wyndham');
INSERT INTO "Desserts" VALUES(5,'Red Velvet Cake',7.99,20,'Intercontinental');
CREATE TABLE Salads(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Price REAL ,
    Quantity INTEGER ,
    BuffetName TEXT
);
INSERT INTO "Salads" VALUES(1,'Greek Salad',9.99,30,'Hilton');
INSERT INTO "Salads" VALUES(2,'Caesar Salad',8.99,25,'Marriott');
INSERT INTO "Salads" VALUES(3,'Mixed Greens Salad',7.99,20,'Hyatt');
INSERT INTO "Salads" VALUES(4,'Veggie Delight',6.99,15,'Wyndham');
INSERT INTO "Salads" VALUES(5,'Quinoa Salad',7.99,20,'Intercontinental');
CREATE TABLE Soups(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Price REAL ,
    Quantity INTEGER ,
    BuffetName TEXT
);
INSERT INTO "Soups" VALUES(1,'Tomato Bisque',5.99,25,'Hilton');
INSERT INTO "Soups" VALUES(2,'Chicken Noodle',4.99,30,'Marriott');
INSERT INTO "Soups" VALUES(3,'Minestrone',6.99,20,'Hyatt');
INSERT INTO "Soups" VALUES(4,'Lobster Bisque',9.99,15,'Wyndham');
INSERT INTO "Soups" VALUES(5,'Vegetable Beef Barley',6.99,20,'Intercontinental');
CREATE TABLE Staffs(
    id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Position TEXT,
    HireDate TEXT
);
INSERT INTO "Staffs" VALUES(1,'John','Doe','Manager','2021-05-01');
INSERT INTO "Staffs" VALUES(2,'Jane','Smith','Waiter','2021-06-01');
INSERT INTO "Staffs" VALUES(3,'Alice','Johnson','Host','2021-07-01');
INSERT INTO "Staffs" VALUES(4,'Bob','Williams','Chef','2021-08-01');
INSERT INTO "Staffs" VALUES(5,'Emily','Davis','Busser','2021-09-01');
COMMIT;
