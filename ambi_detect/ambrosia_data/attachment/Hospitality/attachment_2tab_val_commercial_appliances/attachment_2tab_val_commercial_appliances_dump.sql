BEGIN TRANSACTION;
CREATE TABLE Apartments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    type TEXT ,
    city TEXT
);
INSERT INTO "Apartments" VALUES(1,'High-end Residential','Residential','New York');
INSERT INTO "Apartments" VALUES(2,'Luxury Hotel','Hotel','Los Angeles');
CREATE TABLE Blender(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    apartment_id INTEGER,
    model TEXT,
    price REAL,
    FOREIGN KEY (apartment_id) REFERENCES Apartments(id) ON DELETE CASCADE
);
INSERT INTO "Blender" VALUES(1,1,'Model P',150.0);
INSERT INTO "Blender" VALUES(2,2,'Model Q',120.0);
CREATE TABLE Espresso_Machine(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    apartment_id INTEGER,
    model TEXT,
    price REAL,
    FOREIGN KEY (apartment_id) REFERENCES Apartments(id) ON DELETE CASCADE
);
INSERT INTO "Espresso_Machine" VALUES(1,1,'Model A',300.0);
INSERT INTO "Espresso_Machine" VALUES(2,2,'Model B',250.0);
CREATE TABLE Guests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT
);
INSERT INTO "Guests" VALUES(1,'John','Doe','john@example.com','1234567890');
INSERT INTO "Guests" VALUES(2,'Jane','Smith','jane@example.com','0987654321');
CREATE TABLE Orders(
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_id INTEGER,
    room_id INTEGER,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount REAL,
    status TEXT ,
    FOREIGN KEY (guest_id) REFERENCES Guests(id),
    FOREIGN KEY (room_id) REFERENCES Rooms(id)
);
INSERT INTO "Orders" VALUES(1,1,1,'2024-03-13 17:20:34',150.0,'Pending');
INSERT INTO "Orders" VALUES(2,2,2,'2024-03-13 17:20:34',200.0,'Completed');
CREATE TABLE Rooms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    apartment_id INTEGER,
    number INTEGER,
    floor_number INTEGER,
    FOREIGN KEY (apartment_id) REFERENCES Apartments(id) ON DELETE CASCADE
);
INSERT INTO "Rooms" VALUES(1,1,101,2);
INSERT INTO "Rooms" VALUES(2,2,301,7);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Apartments',2);
INSERT INTO "sqlite_sequence" VALUES('Rooms',2);
INSERT INTO "sqlite_sequence" VALUES('Guests',2);
INSERT INTO "sqlite_sequence" VALUES('Orders',2);
INSERT INTO "sqlite_sequence" VALUES('Espresso_Machine',2);
INSERT INTO "sqlite_sequence" VALUES('Blender',2);
COMMIT;
