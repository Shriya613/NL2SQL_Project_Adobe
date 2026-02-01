BEGIN TRANSACTION;
CREATE TABLE MenuItems(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ItemName TEXT,
    Description TEXT,
    Price DECIMAL(7,2),
    RestaurantId INTEGER,
    FOREIGN KEY(RestaurantId) REFERENCES Restaurants(id)
);
INSERT INTO "MenuItems" VALUES(1,'T-bone Steak','A delicious T-bone steak with garlic butter.',35.99,1);
INSERT INTO "MenuItems" VALUES(2,'Shrimp Scampi','Grilled shrimps sauteed with white wine, garlic and parsley.',20.99,3);
INSERT INTO "MenuItems" VALUES(3,'Garlic Bread','Crusty bread baked with lots of freshly minced garlic.',6.99,4);
INSERT INTO "MenuItems" VALUES(4,'Mixed Greens Salad','A refreshing salad with a variety of mixed greens, cherry tomatoes, cucumber and feta cheese.',8.99,2);
CREATE TABLE Reservations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerName TEXT,
    ContactNumber TEXT,
    NumInParty INTEGER,
    Date DATE,
    Time TIME,
    RestaurantId INTEGER,
    FOREIGN KEY(RestaurantId) REFERENCES Restaurants(id)
);
INSERT INTO "Reservations" VALUES(1,'Alex Johnson','+1234567890',4,'2022-06-15','19:00:00',1);
INSERT INTO "Reservations" VALUES(2,'Emily Smith','+1234567891',2,'2022-06-17','20:00:00',2);
INSERT INTO "Reservations" VALUES(3,'Chris Brown','+1234567892',6,'2022-06-18','18:00:00',3);
INSERT INTO "Reservations" VALUES(4,'Kelly Clarkson','+1234567893',8,'2022-06-19','17:00:00',4);
CREATE TABLE Restaurants(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Type TEXT ,
    Year_of_Establishment INT 
);
INSERT INTO "Restaurants" VALUES(1,'Fine Steaks','Steakhouse',1980);
INSERT INTO "Restaurants" VALUES(2,'The Rib House','Steakhouse',2005);
INSERT INTO "Restaurants" VALUES(3,'Surfside Seafood','Seafood Restaurant',1980);
INSERT INTO "Restaurants" VALUES(4,'Coastal Catch','Seafood Restaurant',2010);
CREATE TABLE Staff(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Position TEXT,
    RestaurantId INTEGER,
    FOREIGN KEY(RestaurantId) REFERENCES Restaurants(id)
);
INSERT INTO "Staff" VALUES(1,'John','Doe','Manager',1);
INSERT INTO "Staff" VALUES(2,'Jane','Smith','Waiter',1);
INSERT INTO "Staff" VALUES(3,'Bob','Johnson','Chef',2);
INSERT INTO "Staff" VALUES(4,'Alice','Williams','Hostess',3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Restaurants',4);
INSERT INTO "sqlite_sequence" VALUES('Staff',4);
INSERT INTO "sqlite_sequence" VALUES('MenuItems',4);
INSERT INTO "sqlite_sequence" VALUES('Reservations',4);
COMMIT;
