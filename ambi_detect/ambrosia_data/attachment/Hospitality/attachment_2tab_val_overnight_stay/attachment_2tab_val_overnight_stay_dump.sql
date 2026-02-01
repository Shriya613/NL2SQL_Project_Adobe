BEGIN TRANSACTION;
CREATE TABLE Employees(
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    role TEXT,
    department TEXT,
    hire_date DATE);
INSERT INTO "Employees" VALUES(1,'Mike Brown','Housekeeping Supervisor','Housekeeping','2021-07-15');
INSERT INTO "Employees" VALUES(2,'Emily White','Front Desk Manager','Reception','2021-06-01');
INSERT INTO "Employees" VALUES(3,'David Green','Maintenance Technician','Engineering','2021-05-15');
CREATE TABLE FoodMenu(
    menu_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT,
    description TEXT,
    price REAL);
INSERT INTO "FoodMenu" VALUES(1,'Steak','Grilled steak served with vegetables.',25.0);
INSERT INTO "FoodMenu" VALUES(2,'Seafood Platter','Assorted seafood with lemon butter sauce.',30.0);
INSERT INTO "FoodMenu" VALUES(3,'Veggie Pizza','Thick crust pizza loaded with vegetables.',15.0);
CREATE TABLE GuestReviews(
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_name TEXT,
    rating INTEGER,
    comments TEXT);
INSERT INTO "GuestReviews" VALUES(1,'Tom Lee',5,'Excellent service and clean rooms.');
INSERT INTO "GuestReviews" VALUES(2,'Karen Kim',4,'Nice location but could use some updates.');
INSERT INTO "GuestReviews" VALUES(3,'Paul Wang',3,'Good experience overall but had a minor issue with WiFi.');
CREATE TABLE Guestrooms(
    id INTEGER PRIMARY KEY,
    "room" TEXT,
    floor_no INTEGER,
    capacity INTEGER,
    amenities TEXT);
INSERT INTO "Guestrooms" VALUES(1,'G001',2,3,'Air Conditioning');
INSERT INTO "Guestrooms" VALUES(2,'G002',3,4,'WiFi Access');
INSERT INTO "Guestrooms" VALUES(3,'G003',1,2,'Parking Facility');
CREATE TABLE Reservations(
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_name TEXT,
    checkin DATE,
    checkout DATE,
    room_type TEXT,
    total_cost REAL);
INSERT INTO "Reservations" VALUES(1,'John Doe','2022-09-01','2022-09-07','Guestroom',500.0);
INSERT INTO "Reservations" VALUES(2,'Jane Smith','2022-09-15','2022-09-20','Guestroom',700.0);
INSERT INTO "Reservations" VALUES(3,'Alice Johnson','2022-09-25','2022-09-30','Suite',1000.0);
CREATE TABLE Suites(
    id INTEGER PRIMARY KEY,
    suite_name TEXT,
    area_sqft INTEGER,
    occupancy INTEGER,
    amenities TEXT);
INSERT INTO "Suites" VALUES(1,'Presidential Suite',2000,6,'Air Conditioning');
INSERT INTO "Suites" VALUES(2,'Deluxe Suite',800,4,'WiFi Access');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Reservations',3);
INSERT INTO "sqlite_sequence" VALUES('Employees',3);
INSERT INTO "sqlite_sequence" VALUES('GuestReviews',3);
INSERT INTO "sqlite_sequence" VALUES('FoodMenu',3);
COMMIT;
