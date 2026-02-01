BEGIN TRANSACTION;
CREATE TABLE Bookings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    GuestId INTEGER,
    StartDate DATE,
    EndDate DATE,
    AccommodationType TEXT,
    FOREIGN KEY(GuestId) REFERENCES Guests(id)
);
INSERT INTO "Bookings" VALUES(1,1,'2022-01-01','2022-01-07','Chalet');
INSERT INTO "Bookings" VALUES(2,2,'2022-01-01','2022-01-07','Cabin');
INSERT INTO "Bookings" VALUES(3,3,'2022-01-15','2022-01-22','Chalet');
INSERT INTO "Bookings" VALUES(4,4,'2022-01-15','2022-01-22','Cabin');
CREATE TABLE Cabins(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Accommodation TEXT ,
    Price_Per_Night REAL,
    Capacity INTEGER
);
INSERT INTO "Cabins" VALUES(1,'Jersey','Cabin',200.0,2);
INSERT INTO "Cabins" VALUES(2,'Lancaster','Cabin',180.0,4);
INSERT INTO "Cabins" VALUES(3,'Leicester','Cabin',300.0,8);
INSERT INTO "Cabins" VALUES(4,'Newcastle','Chalet',200.0,6);
INSERT INTO "Cabins" VALUES(5,'Nottingham','Chalet',190.0,4);
INSERT INTO "Cabins" VALUES(6,'Shrewsbury','Chalet',220.0,6);
INSERT INTO "Cabins" VALUES(7,'Somerset','Chalet',250.0,10);
INSERT INTO "Cabins" VALUES(8,'Staffordshire','Cabin',270.0,8);
CREATE TABLE Chalets(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Accommodation TEXT ,
    Price_Per_Night REAL,
    Capacity INTEGER
);
INSERT INTO "Chalets" VALUES(1,'Ashford','Chalet',200.0,6);
INSERT INTO "Chalets" VALUES(2,'Beechwood','Chalet',180.0,4);
INSERT INTO "Chalets" VALUES(3,'Canterbury','Chalet',300.0,8);
INSERT INTO "Chalets" VALUES(4,'Durham','Cabin',200.0,2);
INSERT INTO "Chalets" VALUES(5,'Exeter','Cabin',190.0,4);
INSERT INTO "Chalets" VALUES(6,'Gloucester','Cabin',220.0,6);
INSERT INTO "Chalets" VALUES(7,'Hampshire','Chalet',250.0,10);
INSERT INTO "Chalets" VALUES(8,'IsleOfWight','Cabin',270.0,8);
CREATE TABLE Guests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    ContactNumber TEXT,
    EmailAddress TEXT UNIQUE
);
INSERT INTO "Guests" VALUES(1,'John','Smith','+123456789','johnsmith@example.com');
INSERT INTO "Guests" VALUES(2,'Jane','Doe','+987654321','janedoe@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Chalets',8);
INSERT INTO "sqlite_sequence" VALUES('Cabins',8);
INSERT INTO "sqlite_sequence" VALUES('Guests',2);
INSERT INTO "sqlite_sequence" VALUES('Bookings',4);
COMMIT;
