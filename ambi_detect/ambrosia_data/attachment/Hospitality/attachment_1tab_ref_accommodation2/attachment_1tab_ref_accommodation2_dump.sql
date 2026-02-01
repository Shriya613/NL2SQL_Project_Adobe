BEGIN TRANSACTION;
CREATE TABLE Accomodations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    type TEXT 
);
INSERT INTO "Accomodations" VALUES(1,'Mountain Chalet','Chalet');
INSERT INTO "Accomodations" VALUES(2,'Lakeside Cabin','Cabin');
INSERT INTO "Accomodations" VALUES(3,'River View Lodge','Chalet');
INSERT INTO "Accomodations" VALUES(4,'Forest Hut','Cabin');
INSERT INTO "Accomodations" VALUES(5,'Beach Bungalow','Chalet');
CREATE TABLE Guests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    emailAddress TEXT UNIQUE
);
INSERT INTO "Guests" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Guests" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Guests" VALUES(3,'Michael','Brown','michael.brown@example.com');
INSERT INTO "Guests" VALUES(4,'Emily','White','emily.white@example.com');
INSERT INTO "Guests" VALUES(5,'David','Black','david.black@example.com');
CREATE TABLE Prices(
    accommodationId INTEGER,
    price_per_night REAL,
    FOREIGN KEY(accommodationId) REFERENCES Accomodations(id)
);
INSERT INTO "Prices" VALUES(1,200.0);
INSERT INTO "Prices" VALUES(1,300.0);
INSERT INTO "Prices" VALUES(2,200.0);
INSERT INTO "Prices" VALUES(2,300.0);
INSERT INTO "Prices" VALUES(3,200.0);
CREATE TABLE Reservations(
    reservationId INTEGER PRIMARY KEY AUTOINCREMENT,
    guestId INTEGER,
    accommodationId INTEGER,
    checkInDate DATE,
    checkOutDate DATE,
    totalCost REAL,
    FOREIGN KEY(guestId) REFERENCES Guests(id),
    FOREIGN KEY(accommodationId) REFERENCES Accomodations(id)
);
INSERT INTO "Reservations" VALUES(1,1,1,'2022-10-01','2022-10-07',1200.0);
INSERT INTO "Reservations" VALUES(2,2,1,'2022-10-14','2022-10-21',1800.0);
INSERT INTO "Reservations" VALUES(3,3,2,'2022-10-01','2022-10-07',1600.0);
INSERT INTO "Reservations" VALUES(4,4,2,'2022-10-14','2022-10-21',2000.0);
INSERT INTO "Reservations" VALUES(5,5,3,'2022-10-01','2022-10-07',900.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Accomodations',5);
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('Reservations',5);
COMMIT;
