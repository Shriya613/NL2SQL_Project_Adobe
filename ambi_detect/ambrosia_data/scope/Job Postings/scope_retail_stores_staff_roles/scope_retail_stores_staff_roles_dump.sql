BEGIN TRANSACTION;
CREATE TABLE Retail_Stores (
    store_id INTEGER PRIMARY KEY AUTOINCREMENT,
    store_name TEXT,
    store_location TEXT
);
INSERT INTO "Retail_Stores" VALUES(1,'Store A','New York City');
INSERT INTO "Retail_Stores" VALUES(2,'Store B','Los Angeles');
INSERT INTO "Retail_Stores" VALUES(3,'Store C','Chicago');
INSERT INTO "Retail_Stores" VALUES(4,'Store D','Houston');
INSERT INTO "Retail_Stores" VALUES(5,'Store E','Miami');
CREATE TABLE Retail_Stores_Staff_Roles (
    retail_store_id INTEGER,
    staff_role_id INTEGER,
    FOREIGN KEY(retail_store_id) REFERENCES Retail_Stores(store_id),
    FOREIGN KEY(staff_role_id) REFERENCES Staff_Roles(role_id)
);
INSERT INTO "Retail_Stores_Staff_Roles" VALUES(1,1);
INSERT INTO "Retail_Stores_Staff_Roles" VALUES(2,1);
INSERT INTO "Retail_Stores_Staff_Roles" VALUES(3,1);
INSERT INTO "Retail_Stores_Staff_Roles" VALUES(4,1);
INSERT INTO "Retail_Stores_Staff_Roles" VALUES(5,1);
INSERT INTO "Retail_Stores_Staff_Roles" VALUES(5,2);
CREATE TABLE Staff_Roles (
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_title TEXT
);
INSERT INTO "Staff_Roles" VALUES(1,'Sales Associate');
INSERT INTO "Staff_Roles" VALUES(2,'Manager');
INSERT INTO "Staff_Roles" VALUES(3,'Assistant Manager');
INSERT INTO "Staff_Roles" VALUES(4,'Cashier');
INSERT INTO "Staff_Roles" VALUES(5,'Stock Clerk');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Staff_Roles',5);
INSERT INTO "sqlite_sequence" VALUES('Retail_Stores',5);
COMMIT;
