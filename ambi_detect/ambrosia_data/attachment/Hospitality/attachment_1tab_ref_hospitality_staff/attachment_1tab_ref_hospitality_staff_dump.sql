BEGIN TRANSACTION;
CREATE TABLE Bookings (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    GuestName TEXT,
    CheckInDate DATE,
    CheckOutDate DATE,
    RoomID INTEGER,
    FOREIGN KEY(RoomID) REFERENCES Rooms(ID));
INSERT INTO "Bookings" VALUES(1,'Alice Cooper','2023-06-01','2023-06-07',101);
INSERT INTO "Bookings" VALUES(2,'Bob Marley','2023-06-08','2023-06-14',102);
INSERT INTO "Bookings" VALUES(3,'Carol King','2023-06-15','2023-06-21',103);
INSERT INTO "Bookings" VALUES(4,'Daniel Ricciardo','2023-06-22','2023-06-29',104);
INSERT INTO "Bookings" VALUES(5,'Eva Mendes','2023-06-30','2023-07-07',105);
CREATE TABLE Languages (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Language TEXT UNIQUE);
INSERT INTO "Languages" VALUES(1,'Spanish');
INSERT INTO "Languages" VALUES(2,'French');
INSERT INTO "Languages" VALUES(3,'German');
INSERT INTO "Languages" VALUES(4,'Italian');
INSERT INTO "Languages" VALUES(5,'Japanese');
CREATE TABLE Rooms (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    RoomNumber INTEGER);
INSERT INTO "Rooms" VALUES(1,101);
INSERT INTO "Rooms" VALUES(2,102);
INSERT INTO "Rooms" VALUES(3,103);
INSERT INTO "Rooms" VALUES(4,104);
INSERT INTO "Rooms" VALUES(5,105);
CREATE TABLE Staff (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Position TEXT,
    LanguageID INTEGER,
    FOREIGN KEY(LanguageID) REFERENCES Languages(ID));
INSERT INTO "Staff" VALUES(1,'John Doe','Chef',1);
INSERT INTO "Staff" VALUES(2,'Mike Smith','Chef',3);
INSERT INTO "Staff" VALUES(3,'Emily Johnson','Server',1);
INSERT INTO "Staff" VALUES(4,'David Brown','Server',4);
INSERT INTO "Staff" VALUES(5,'Anna Davis','Manager',2);
INSERT INTO "Staff" VALUES(6,'Peter Miller','Bartender',5);
CREATE TABLE StaffRooms (
    StaffID INTEGER,
    RoomID INTEGER,
    WorkDays TEXT,
    PRIMARY KEY(StaffID, RoomID),
    FOREIGN KEY(StaffID) REFERENCES Staff(ID),
    FOREIGN KEY(RoomID) REFERENCES Rooms(ID));
INSERT INTO "StaffRooms" VALUES(5,101,'Monday,Wednesday');
INSERT INTO "StaffRooms" VALUES(6,102,'Tuesday,Thursday');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Languages',5);
INSERT INTO "sqlite_sequence" VALUES('Staff',6);
INSERT INTO "sqlite_sequence" VALUES('Rooms',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
COMMIT;
