BEGIN TRANSACTION;
CREATE TABLE Craft_Services(
    ServiceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    Price REAL
);
INSERT INTO "Craft_Services" VALUES(1,'Fresh Fruit Platter',20.99);
INSERT INTO "Craft_Services" VALUES(2,'Assorted Cookies',18.75);
INSERT INTO "Craft_Services" VALUES(3,'Mixed Nuts Bowl',30.66);
INSERT INTO "Craft_Services" VALUES(4,'Tea & Coffee Station',15.0);
INSERT INTO "Craft_Services" VALUES(5,'Water Bottles',5.0);
CREATE TABLE Craft_Services_Menus(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ServiceID INTEGER,
    MenuID INTEGER,
    FOREIGN KEY (ServiceID) REFERENCES Craft_Services(ServiceID),
    FOREIGN KEY (MenuID) REFERENCES Menus(MenuID)
);
INSERT INTO "Craft_Services_Menus" VALUES(1,1,1);
INSERT INTO "Craft_Services_Menus" VALUES(2,2,1);
INSERT INTO "Craft_Services_Menus" VALUES(3,3,1);
INSERT INTO "Craft_Services_Menus" VALUES(4,4,1);
INSERT INTO "Craft_Services_Menus" VALUES(5,5,1);
INSERT INTO "Craft_Services_Menus" VALUES(6,1,4);
CREATE TABLE Menus(
    MenuID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Menus" VALUES(1,'Vegan');
INSERT INTO "Menus" VALUES(2,'Non-vegan');
INSERT INTO "Menus" VALUES(3,'Kid Friendly');
INSERT INTO "Menus" VALUES(4,'Gluten Free');
INSERT INTO "Menus" VALUES(5,'Desserts');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Menus',5);
INSERT INTO "sqlite_sequence" VALUES('Craft_Services',5);
INSERT INTO "sqlite_sequence" VALUES('Craft_Services_Menus',6);
COMMIT;
