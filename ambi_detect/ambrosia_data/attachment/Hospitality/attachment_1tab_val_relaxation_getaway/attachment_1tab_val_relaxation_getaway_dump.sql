BEGIN TRANSACTION;
CREATE TABLE Guests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    phoneNumber TEXT);
INSERT INTO "Guests" VALUES(1,'John','Doe','+123456789');
INSERT INTO "Guests" VALUES(2,'Jane','Smith','+987654321');
CREATE TABLE "Relaxation_spot"(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
    type TEXT ,
    price REAL);
INSERT INTO "Relaxation_spot" VALUES(1,'Spot A','Day Spa',150.0);
INSERT INTO "Relaxation_spot" VALUES(2,'Spot B','Day Spa',200.0);
INSERT INTO "Relaxation_spot" VALUES(3,'Spot C','Wellness Retreat',150.0);
INSERT INTO "Relaxation_spot" VALUES(4,'Spot D','Wellness Retreat',250.0);
CREATE TABLE Reservations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guestId INTEGER,
    roomId INTEGER,
    checkInDate TEXT,
    checkOutDate TEXT,
    FOREIGN KEY (guestId) REFERENCES Guests(id));
INSERT INTO "Reservations" VALUES(1,1,101,'2023-01-01','2023-01-05');
INSERT INTO "Reservations" VALUES(2,2,102,'2023-01-06','2023-01-10');
CREATE TABLE Rooms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roomNumber INTEGER,
    capacity INTEGER,
    pricePerNight REAL);
INSERT INTO "Rooms" VALUES(1,101,2,100.0);
INSERT INTO "Rooms" VALUES(2,102,4,150.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Guests',2);
INSERT INTO "sqlite_sequence" VALUES('Rooms',2);
INSERT INTO "sqlite_sequence" VALUES('Reservations',2);
INSERT INTO "sqlite_sequence" VALUES('Relaxation_spot',4);
COMMIT;
