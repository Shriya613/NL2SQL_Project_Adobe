BEGIN TRANSACTION;
CREATE TABLE Kitchen_Roles (
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name VARCHAR(255)
);
INSERT INTO "Kitchen_Roles" VALUES(1,'Chef');
INSERT INTO "Kitchen_Roles" VALUES(2,'Line Cook');
INSERT INTO "Kitchen_Roles" VALUES(3,'Prep Cook');
INSERT INTO "Kitchen_Roles" VALUES(4,'Dishwasher');
INSERT INTO "Kitchen_Roles" VALUES(5,'Host/Hostess');
CREATE TABLE Restaurants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    phone_number TEXT
);
INSERT INTO "Restaurants" VALUES(1,'Pizza Palace','123 Main St','555-0100');
INSERT INTO "Restaurants" VALUES(2,'Burger Joint','456 Elm St','555-0101');
INSERT INTO "Restaurants" VALUES(3,'Taco Town','789 Oak St','555-0102');
INSERT INTO "Restaurants" VALUES(4,'Seafood Shack','123 Pine St','555-0103');
INSERT INTO "Restaurants" VALUES(5,'French Bistro','456 Maple St','555-0104');
CREATE TABLE Restaurants_Kitchen_Roles (
    restaurant_id INTEGER,
    kitchen_role_id INTEGER,
    FOREIGN KEY (restaurant_id) REFERENCES Restaurants(id),
    FOREIGN KEY (kitchen_role_id) REFERENCES Kitchen_Roles(role_id)
);
INSERT INTO "Restaurants_Kitchen_Roles" VALUES(1,1);
INSERT INTO "Restaurants_Kitchen_Roles" VALUES(2,1);
INSERT INTO "Restaurants_Kitchen_Roles" VALUES(3,1);
INSERT INTO "Restaurants_Kitchen_Roles" VALUES(4,1);
INSERT INTO "Restaurants_Kitchen_Roles" VALUES(5,1);
INSERT INTO "Restaurants_Kitchen_Roles" VALUES(5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Restaurants',5);
INSERT INTO "sqlite_sequence" VALUES('Kitchen_Roles',5);
COMMIT;
