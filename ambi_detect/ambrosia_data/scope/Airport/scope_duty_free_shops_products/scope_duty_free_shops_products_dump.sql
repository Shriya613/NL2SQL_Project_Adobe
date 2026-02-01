BEGIN TRANSACTION;
CREATE TABLE Duty_Free_Shops (
    shop_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Duty_Free_Shops" VALUES(1,'Shop1','Terminal A');
INSERT INTO "Duty_Free_Shops" VALUES(2,'Shop2','Terminal B');
INSERT INTO "Duty_Free_Shops" VALUES(3,'Shop3','Terminal C');
INSERT INTO "Duty_Free_Shops" VALUES(4,'Shop4','Terminal D');
INSERT INTO "Duty_Free_Shops" VALUES(5,'Shop5','Terminal E');
CREATE TABLE Duty_Free_Shops_Products (
    shop_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    FOREIGN KEY(shop_id) REFERENCES Duty_Free_Shops(shop_id),
    FOREIGN KEY(product_id) REFERENCES Products(product_id),
    PRIMARY KEY(shop_id, product_id)
);
INSERT INTO "Duty_Free_Shops_Products" VALUES(1,1,10);
INSERT INTO "Duty_Free_Shops_Products" VALUES(1,2,20);
INSERT INTO "Duty_Free_Shops_Products" VALUES(2,3,15);
INSERT INTO "Duty_Free_Shops_Products" VALUES(2,4,30);
INSERT INTO "Duty_Free_Shops_Products" VALUES(3,1,5);
INSERT INTO "Duty_Free_Shops_Products" VALUES(3,2,10);
INSERT INTO "Duty_Free_Shops_Products" VALUES(4,1,15);
INSERT INTO "Duty_Free_Shops_Products" VALUES(4,3,20);
INSERT INTO "Duty_Free_Shops_Products" VALUES(5,2,25);
INSERT INTO "Duty_Free_Shops_Products" VALUES(5,4,30);
INSERT INTO "Duty_Free_Shops_Products" VALUES(2,1,15);
INSERT INTO "Duty_Free_Shops_Products" VALUES(5,1,25);
CREATE TABLE Products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    price REAL
);
INSERT INTO "Products" VALUES(1,'Perfume',90.78);
INSERT INTO "Products" VALUES(2,'Cosmetics',65.43);
INSERT INTO "Products" VALUES(3,'Electronics',250.89);
INSERT INTO "Products" VALUES(4,'Fashion Accessories',150.99);
INSERT INTO "Products" VALUES(5,'Toys',20.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Duty_Free_Shops',5);
INSERT INTO "sqlite_sequence" VALUES('Products',5);
COMMIT;
