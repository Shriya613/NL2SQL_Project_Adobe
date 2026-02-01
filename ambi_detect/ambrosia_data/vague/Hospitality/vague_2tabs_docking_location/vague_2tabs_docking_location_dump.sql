BEGIN TRANSACTION;
CREATE TABLE Cities (
    name TEXT,
    country TEXT);
INSERT INTO "Cities" VALUES('Miami','USA');
INSERT INTO "Cities" VALUES('Barcelona','Spain');
INSERT INTO "Cities" VALUES('Venice','Italy');
INSERT INTO "Cities" VALUES('Hong Kong','China');
INSERT INTO "Cities" VALUES('Singapore','Singapore');
CREATE TABLE CruiseShips (
    port TEXT,
    city TEXT,
    shipName TEXT,
    maxCapacity INT,
    FOREIGN KEY(port) REFERENCES Ports(name),
    FOREIGN KEY(city) REFERENCES Cities(name));
INSERT INTO "CruiseShips" VALUES('PortMiami','Miami','Queen Mary II',2620);
INSERT INTO "CruiseShips" VALUES('World Trade Center Barcelona','Barcelona','Allure of the Seas',5400);
INSERT INTO "CruiseShips" VALUES('Terminal Passeggeri di Venezia','Venice','Seabourn Odyssey',458);
INSERT INTO "CruiseShips" VALUES('Kowloon Dockyard','Hong Kong','Golden Era',1400);
INSERT INTO "CruiseShips" VALUES('HarbourFront Centre','Singapore','Mariner of the Seas',3804);
CREATE TABLE Payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reservationId INTEGER,
    amountPaid DECIMAL(8,2),
    paymentMethod TEXT,
    transactionTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(reservationId) REFERENCES Reservations(id));
INSERT INTO "Payments" VALUES(1,1,300,'Credit Card','2022-06-01 10:00:00');
INSERT INTO "Payments" VALUES(2,2,600,'Debit Card','2022-06-05 14:00:00');
INSERT INTO "Payments" VALUES(3,3,900,'PayPal','2022-06-10 08:00:00');
INSERT INTO "Payments" VALUES(4,4,1200,'Wire Transfer','2022-06-14 12:00:00');
INSERT INTO "Payments" VALUES(5,5,1500,'Cash','2022-06-18 16:00:00');
CREATE TABLE Ports (
    name TEXT,
    country TEXT);
INSERT INTO "Ports" VALUES('PortMiami','USA');
INSERT INTO "Ports" VALUES('World Trade Center Barcelona','Spain');
INSERT INTO "Ports" VALUES('Terminal Passeggeri di Venezia','Italy');
INSERT INTO "Ports" VALUES('Kowloon Dockyard','China');
INSERT INTO "Ports" VALUES('HarbourFront Centre','Singapore');
CREATE TABLE Reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customerId INTEGER,
    roomTypeId INTEGER,
    checkInDate DATE,
    checkOutDate DATE,
    FOREIGN KEY(customerId) REFERENCES Customers(id),
    FOREIGN KEY(roomTypeId) REFERENCES RoomTypes(id));
INSERT INTO "Reservations" VALUES(1,1,1,'2022-06-01','2022-06-05');
INSERT INTO "Reservations" VALUES(2,2,2,'2022-06-05','2022-06-10');
INSERT INTO "Reservations" VALUES(3,3,3,'2022-06-10','2022-06-14');
INSERT INTO "Reservations" VALUES(4,4,4,'2022-06-14','2022-06-18');
INSERT INTO "Reservations" VALUES(5,5,5,'2022-06-18','2022-06-22');
CREATE TABLE RoomTypes (
    id INTEGER PRIMARY KEY,
    typeName TEXT,
    capacity INT,
    pricePerNight REAL);
INSERT INTO "RoomTypes" VALUES(1,'Standard',3,100.99);
INSERT INTO "RoomTypes" VALUES(2,'Deluxe',4,150.99);
INSERT INTO "RoomTypes" VALUES(3,'Suite',6,200.99);
INSERT INTO "RoomTypes" VALUES(4,'Family Suite',7,250.99);
INSERT INTO "RoomTypes" VALUES(5,'Executive Suite',8,300.99);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Reservations',5);
INSERT INTO "sqlite_sequence" VALUES('Payments',5);
COMMIT;
