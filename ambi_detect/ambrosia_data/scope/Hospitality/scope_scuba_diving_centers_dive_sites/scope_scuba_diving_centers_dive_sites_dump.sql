BEGIN TRANSACTION;
CREATE TABLE Bookings (
    booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    scuba_center_id INTEGER,
    dive_site_id INTEGER,
    date DATE,
    time TIME,
    FOREIGN KEY(customer_id) REFERENCES Customers(id),
    FOREIGN KEY(scuba_center_id) REFERENCES Scuba_Diving_Centers(id),
    FOREIGN KEY(dive_site_id) REFERENCES Dive_Sites(id)
);
INSERT INTO "Bookings" VALUES(1,1,1,1,'2023-07-16','10:00');
INSERT INTO "Bookings" VALUES(2,2,2,2,'2023-08-09','11:00');
INSERT INTO "Bookings" VALUES(3,3,3,3,'2023-09-17','12:00');
INSERT INTO "Bookings" VALUES(4,4,4,4,'2023-10-23','13:00');
INSERT INTO "Bookings" VALUES(5,5,5,5,'2023-11-30','14:00');
INSERT INTO "Bookings" VALUES(6,2,2,1,'2023-08-09','11:00');
INSERT INTO "Bookings" VALUES(7,3,3,1,'2023-09-17','12:00');
INSERT INTO "Bookings" VALUES(8,4,4,1,'2023-10-23','13:00');
INSERT INTO "Bookings" VALUES(9,5,5,1,'2023-11-30','14:00');
CREATE TABLE Customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT
);
INSERT INTO "Customers" VALUES(1,'John','Smith','johnsmith@example.com');
INSERT INTO "Customers" VALUES(2,'Jane','Doe','janedoe@example.com');
INSERT INTO "Customers" VALUES(3,'Alice','King','aliceking@example.com');
INSERT INTO "Customers" VALUES(4,'Bob','Johnson','bobjohnson@example.com');
INSERT INTO "Customers" VALUES(5,'Charlie','Williams','charliewilliams@example.com');
CREATE TABLE Dive_Sites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_name TEXT,
    type TEXT 
);
INSERT INTO "Dive_Sites" VALUES(1,'Great Barrier Reef','Coral_Reefs');
INSERT INTO "Dive_Sites" VALUES(2,'Shipwreck Cove','Wrecks');
INSERT INTO "Dive_Sites" VALUES(3,'Shark Alley','Marine Life');
INSERT INTO "Dive_Sites" VALUES(4,'Turtle Bay','Marine Life');
INSERT INTO "Dive_Sites" VALUES(5,'The Blue Hole','Caverns');
CREATE TABLE Scuba_Diving_Centers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Scuba_Diving_Centers" VALUES(1,'Blue Wave','Cancun');
INSERT INTO "Scuba_Diving_Centers" VALUES(2,'Sea Explorer','Mauritius');
INSERT INTO "Scuba_Diving_Centers" VALUES(3,'Underwater World','Bali');
INSERT INTO "Scuba_Diving_Centers" VALUES(4,'Aquatic Adventures','Hawaii');
INSERT INTO "Scuba_Diving_Centers" VALUES(5,'Poseidon Divers','Greece');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Scuba_Diving_Centers',5);
INSERT INTO "sqlite_sequence" VALUES('Dive_Sites',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',9);
COMMIT;
