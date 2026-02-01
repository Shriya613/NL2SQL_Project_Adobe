BEGIN TRANSACTION;
CREATE TABLE Bookings(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    CheckInDate DATETIME,
    CheckOutDate DATETIME,
    RoomId INTEGER,
    GuestId INTEGER,
    FOREIGN KEY(RoomId) REFERENCES Rooms(Id),
    FOREIGN KEY(GuestId) REFERENCES Guests(Id)
);
INSERT INTO "Bookings" VALUES(1,'2022-01-01 10:00:00','2022-01-05 10:00:00',1,1);
INSERT INTO "Bookings" VALUES(2,'2022-01-10 10:00:00','2022-01-14 10:00:00',2,2);
INSERT INTO "Bookings" VALUES(3,'2022-01-15 10:00:00','2022-01-19 10:00:00',3,3);
INSERT INTO "Bookings" VALUES(4,'2022-01-20 10:00:00','2022-01-24 10:00:00',4,4);
INSERT INTO "Bookings" VALUES(5,'2022-01-25 10:00:00','2022-01-29 10:00:00',5,5);
INSERT INTO "Bookings" VALUES(6,'2022-02-01 10:00:00','2022-02-05 10:00:00',6,6);
INSERT INTO "Bookings" VALUES(7,'2022-02-10 10:00:00','2022-02-14 10:00:00',7,7);
INSERT INTO "Bookings" VALUES(8,'2022-02-15 10:00:00','2022-02-19 10:00:00',8,8);
INSERT INTO "Bookings" VALUES(9,'2022-02-20 10:00:00','2022-02-24 10:00:00',9,9);
INSERT INTO "Bookings" VALUES(10,'2022-02-25 10:00:00','2022-02-29 10:00:00',10,10);
CREATE TABLE Guests(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT,
    PhoneNumber TEXT
);
INSERT INTO "Guests" VALUES(1,'John','Doe','john.doe@example.com','+1234567890');
INSERT INTO "Guests" VALUES(2,'Jane','Smith','jane.smith@example.com','+9876543210');
INSERT INTO "Guests" VALUES(3,'Michael','Brown','michael.brown@example.com','+1357924680');
INSERT INTO "Guests" VALUES(4,'Alice','Johnson','alice.johnson@example.com','+1357924680');
INSERT INTO "Guests" VALUES(5,'David','White','david.white@example.com','+1357924680');
CREATE TABLE RoomTypes(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    TypeName TEXT
);
INSERT INTO "RoomTypes" VALUES(1,'Single');
INSERT INTO "RoomTypes" VALUES(2,'Double');
INSERT INTO "RoomTypes" VALUES(3,'Twin');
INSERT INTO "RoomTypes" VALUES(4,'Triple');
INSERT INTO "RoomTypes" VALUES(5,'Suite');
CREATE TABLE Rooms(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    RoomTypeId INTEGER,
    FloorNumber INTEGER,
    FOREIGN KEY(RoomTypeId) REFERENCES RoomTypes(Id)
);
INSERT INTO "Rooms" VALUES(1,1,1);
INSERT INTO "Rooms" VALUES(2,2,1);
INSERT INTO "Rooms" VALUES(3,3,2);
INSERT INTO "Rooms" VALUES(4,4,3);
INSERT INTO "Rooms" VALUES(5,5,3);
INSERT INTO "Rooms" VALUES(6,6,4);
INSERT INTO "Rooms" VALUES(7,7,4);
INSERT INTO "Rooms" VALUES(8,8,5);
INSERT INTO "Rooms" VALUES(9,9,5);
INSERT INTO "Rooms" VALUES(10,10,6);
CREATE TABLE Staff(
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Role TEXT,
    Language TEXT,
    AdditionalColumn1 TEXT,
    AdditionalColumn2 TEXT,
    AdditionalColumn3 TEXT
);
INSERT INTO "Staff" VALUES(1,'Pedro','Chef','Spanish',NULL,NULL,NULL);
INSERT INTO "Staff" VALUES(2,'Emma','Chef','English',NULL,NULL,NULL);
INSERT INTO "Staff" VALUES(3,'Lucas','Server','Spanish',NULL,NULL,NULL);
INSERT INTO "Staff" VALUES(4,'Ava','Server','English',NULL,NULL,NULL);
INSERT INTO "Staff" VALUES(5,'Carlos','Manager','Spanish',NULL,NULL,NULL);
INSERT INTO "Staff" VALUES(6,'Emily','Receptionist','English',NULL,NULL,NULL);
INSERT INTO "Staff" VALUES(7,'Manuel','Maintenance','Spanish',NULL,NULL,NULL);
INSERT INTO "Staff" VALUES(8,'Grace','Housekeeping','English',NULL,NULL,NULL);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('RoomTypes',5);
INSERT INTO "sqlite_sequence" VALUES('Rooms',10);
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',10);
INSERT INTO "sqlite_sequence" VALUES('Staff',8);
COMMIT;
