BEGIN TRANSACTION;
CREATE TABLE AccommodationType(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    typeName TEXT UNIQUE);
INSERT INTO "AccommodationType" VALUES(1,'Standard Pod');
INSERT INTO "AccommodationType" VALUES(2,'Deluxe Pod');
CREATE TABLE Booking(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guestID INTEGER,
    glampingPodId INTEGER,
    checkInDate DATETIME,
    checkOutDate DATETIME,
    totalPrice DECIMAL(10,2),
    FOREIGN KEY (guestID) REFERENCES Guest(id),
    FOREIGN KEY (glampingPodId) REFERENCES GlampingPod(id));
INSERT INTO "Booking" VALUES(1,1,1,'2022-09-01','2022-09-07',500);
INSERT INTO "Booking" VALUES(2,2,2,'2022-09-15','2022-09-21',700);
INSERT INTO "Booking" VALUES(3,3,3,'2022-09-01','2022-09-07',500);
INSERT INTO "Booking" VALUES(4,4,4,'2022-09-15','2022-09-21',700);
INSERT INTO "Booking" VALUES(5,5,5,'2022-09-01','2022-09-07',500);
CREATE TABLE GlampingPod(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "pod" INT,
    capacity INT,
    locationId INTEGER,
    accommodationTypeId INTEGER,
    FOREIGN KEY (locationId) REFERENCES Location(id),
    FOREIGN KEY (accommodationTypeId) REFERENCES AccommodationType(id));
INSERT INTO "GlampingPod" VALUES(1,1,2,1,1);
INSERT INTO "GlampingPod" VALUES(2,2,3,1,1);
INSERT INTO "GlampingPod" VALUES(3,3,2,1,2);
INSERT INTO "GlampingPod" VALUES(4,4,4,2,2);
INSERT INTO "GlampingPod" VALUES(5,5,6,2,2);
CREATE TABLE Guest(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    email TEXT,
    phoneNumber TEXT UNIQUE);
INSERT INTO "Guest" VALUES(1,'John','Doe','john.doe@example.com','+441234567890');
INSERT INTO "Guest" VALUES(2,'Jane','Smith','jane.smith@example.com','+449876543210');
INSERT INTO "Guest" VALUES(3,'Alice','Brown','alice.brown@example.com','+441234567891');
INSERT INTO "Guest" VALUES(4,'Charlie','Black','charlie.black@example.com','+449876543211');
INSERT INTO "Guest" VALUES(5,'David','White','david.white@example.com','+441234567892');
CREATE TABLE Location(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
INSERT INTO "Location" VALUES(1,'Snowdonia');
INSERT INTO "Location" VALUES(2,'Lake District');
CREATE TABLE TreeHouse(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "house" INT,
    capacity INT,
    locationId INTEGER,
    accommodationTypeId INTEGER,
    FOREIGN KEY (locationId) REFERENCES Location(id),
    FOREIGN KEY (accommodationTypeId) REFERENCES AccommodationType(id));
INSERT INTO "TreeHouse" VALUES(1,101,6,1,1);
INSERT INTO "TreeHouse" VALUES(2,102,8,1,1);
INSERT INTO "TreeHouse" VALUES(3,103,6,2,1);
INSERT INTO "TreeHouse" VALUES(4,104,8,2,2);
INSERT INTO "TreeHouse" VALUES(5,105,6,2,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Location',2);
INSERT INTO "sqlite_sequence" VALUES('AccommodationType',2);
INSERT INTO "sqlite_sequence" VALUES('GlampingPod',5);
INSERT INTO "sqlite_sequence" VALUES('TreeHouse',5);
INSERT INTO "sqlite_sequence" VALUES('Booking',5);
INSERT INTO "sqlite_sequence" VALUES('Guest',5);
COMMIT;
