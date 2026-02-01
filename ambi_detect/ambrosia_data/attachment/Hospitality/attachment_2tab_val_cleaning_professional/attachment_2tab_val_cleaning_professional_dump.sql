BEGIN TRANSACTION;
CREATE TABLE GuestRooms(
    room_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hotel_name TEXT,
    room_number INTEGER,
    capacity INTEGER ,
    amenities TEXT
);
INSERT INTO "GuestRooms" VALUES(1,'The Grand Hotel',101,2,'TV, Mini Fridge, Air Conditioner');
INSERT INTO "GuestRooms" VALUES(2,'Ritz Carlton',301,3,'TV, Mini Fridge, Air Conditioner, Balcony');
CREATE TABLE Housekeeping(
    id INTEGER PRIMARY KEY,
    housekeeper TEXT,
    hotel TEXT,
    shift_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    shift_end TIMESTAMP
);
INSERT INTO "Housekeeping" VALUES(1,'John Doe','The Grand Hotel','2024-03-13 17:44:11',NULL);
INSERT INTO "Housekeeping" VALUES(2,'Jane Smith','Ritz Carlton','2024-03-13 17:44:11',NULL);
CREATE TABLE LaundryService(
    id INTEGER PRIMARY KEY,
    laundry_attendant TEXT,
    hotel TEXT,
    shift_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    shift_end TIMESTAMP
);
INSERT INTO "LaundryService" VALUES(1,'Mary Johnson','The Grand Hotel','2024-03-13 17:44:11',NULL);
INSERT INTO "LaundryService" VALUES(2,'Tom Brown','Four Seasons','2024-03-13 17:44:11',NULL);
CREATE TABLE MaintenanceRequests(
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    status TEXT 
);
INSERT INTO "MaintenanceRequests" VALUES(1,'Leaky faucet in Room 101','2024-03-13 17:44:11',NULL,'Pending');
INSERT INTO "MaintenanceRequests" VALUES(2,'Broken air conditioner in Room 301','2024-03-13 17:44:11','2023-01-02','Resolved');
CREATE TABLE Reservations(
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_name TEXT,
    email TEXT,
    phone_number TEXT,
    checkin_date DATE,
    checkout_date DATE,
    room_id INTEGER,
    FOREIGN KEY (room_id) REFERENCES GuestRooms(room_id)
);
INSERT INTO "Reservations" VALUES(1,'Alice Jones','alice@example.com','+1 9876543210','2023-01-01','2023-01-05',1);
INSERT INTO "Reservations" VALUES(2,'Bob Smith','bob@example.com','+1 5556667777','2023-02-01','2023-02-05',2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('GuestRooms',2);
INSERT INTO "sqlite_sequence" VALUES('Reservations',2);
INSERT INTO "sqlite_sequence" VALUES('MaintenanceRequests',2);
COMMIT;
