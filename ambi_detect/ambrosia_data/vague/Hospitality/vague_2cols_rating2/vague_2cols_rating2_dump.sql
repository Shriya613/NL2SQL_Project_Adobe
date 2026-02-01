BEGIN TRANSACTION;
CREATE TABLE bookings (
    booking_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    checkin DATE,
    checkout DATE,
    total_amount REAL ,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id));
INSERT INTO "bookings" VALUES(1,1,'2023-01-01','2023-01-03',200.0);
INSERT INTO "bookings" VALUES(2,2,'2023-02-01','2023-02-03',250.0);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT UNIQUE);
INSERT INTO "customers" VALUES(1,'John','Doe','john.doe@example.com','123-456-7890');
INSERT INTO "customers" VALUES(2,'Jane','Smith','jane.smith@example.com','098-765-4321');
CREATE TABLE "reviews" (
    review_id INTEGER PRIMARY KEY,
    customer_review TEXT,
    difficulty_level INT, hikes name TEXT);
INSERT INTO "reviews" VALUES(1,'Great experience!',3,'Waterfall loop');
INSERT INTO "reviews" VALUES(2,'Beautiful scenery but challenging.',4,'View point hike');
CREATE TABLE rooms (
    room_id INTEGER PRIMARY KEY,
    room_type TEXT,
    price REAL ,
    available BOOLEAN DEFAULT TRUE);
INSERT INTO "rooms" VALUES(1,'Standard',100.0,1);
INSERT INTO "rooms" VALUES(2,'Deluxe',150.0,1);
COMMIT;
