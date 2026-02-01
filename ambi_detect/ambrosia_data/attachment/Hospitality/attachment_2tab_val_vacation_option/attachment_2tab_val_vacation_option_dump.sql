BEGIN TRANSACTION;
CREATE TABLE All_Inclusive_Resort (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    star_rating INTEGER,
    number_of_rooms INTEGER,
    activities TEXT);
INSERT INTO "All_Inclusive_Resort" VALUES(1,'Paradise Beach','Hawaii',4,150,'Dance Class');
INSERT INTO "All_Inclusive_Resort" VALUES(2,'Sunset Oasis','Mexico',3,200,'Yoga Session');
CREATE TABLE Booking(
    bookingId INTEGER PRIMARY KEY AUTOINCREMENT,
    customerId INTEGER,
    cruiseOrResort BOOLEAN,
    checkinDate DATE,
    checkoutDate DATE,
    FOREIGN KEY(customerId) REFERENCES Customer(id));
INSERT INTO "Booking" VALUES(1,1,1,'2023-06-01','2023-06-07');
INSERT INTO "Booking" VALUES(2,2,0,'2023-06-10','2023-06-20');
CREATE TABLE Cruise_Ship (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    capacity INTEGER,
    destination TEXT,
    cruise_start_date DATE,
    cruise_end_date DATE,
    activities TEXT);
INSERT INTO "Cruise_Ship" VALUES(1,'Majestic Ocean',3000,'Caribbean','2023-06-01','2023-06-07','Dance Class');
INSERT INTO "Cruise_Ship" VALUES(2,'Grand Voyage',2500,'Alaska','2023-05-15','2023-05-28','Bingo Night');
CREATE TABLE Customer(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    phoneNumber CHAR(10),
    preferredLanguage TEXT);
INSERT INTO "Customer" VALUES(1,'John','Smith','johnsmith@example.com','+1234567890','English');
INSERT INTO "Customer" VALUES(2,'Jane','Doe','janedoe@example.com','+9876543210','French');
CREATE TABLE Employee(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    position TEXT,
    yearsOfExperience INTEGER,
    skill TEXT);
INSERT INTO "Employee" VALUES(1,'Tom','Anderson','Receptionist',5,'Customer Service');
INSERT INTO "Employee" VALUES(2,'Laura','Williams','Chef',8,'Cooking');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Cruise_Ship',2);
INSERT INTO "sqlite_sequence" VALUES('All_Inclusive_Resort',2);
INSERT INTO "sqlite_sequence" VALUES('Customer',2);
INSERT INTO "sqlite_sequence" VALUES('Employee',2);
INSERT INTO "sqlite_sequence" VALUES('Booking',2);
COMMIT;
