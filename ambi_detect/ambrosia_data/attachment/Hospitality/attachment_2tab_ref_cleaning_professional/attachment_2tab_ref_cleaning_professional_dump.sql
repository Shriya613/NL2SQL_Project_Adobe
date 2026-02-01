BEGIN TRANSACTION;
CREATE TABLE Hotels(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    phoneNumber TEXT
);
INSERT INTO "Hotels" VALUES(1,'The Grand Hotel','123 Main St','+1234567890');
INSERT INTO "Hotels" VALUES(2,'Park Plaza Inn','456 Market St','+0987654321');
INSERT INTO "Hotels" VALUES(3,'Riverview Suites','789 River Rd','+1112223333');
INSERT INTO "Hotels" VALUES(4,'Sunset Beach Resort','321 Ocean Blvd','+4445556667');
INSERT INTO "Hotels" VALUES(5,'Mountain Retreat Lodge','654 Mountain Way','+7778889990');
CREATE TABLE HouseKeepers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    hireDate DATE,
    salary REAL ,
    hotelId INTEGER, FullName TEXT,
    FOREIGN KEY(hotelId) REFERENCES Hotels(id)
);
INSERT INTO "HouseKeepers" VALUES(1,'john.doe@example.com','2021-01-01',35000.0,1,'John Doe');
INSERT INTO "HouseKeepers" VALUES(2,'jane.smith@example.com','2021-02-01',38000.0,2,'Jane Smith');
INSERT INTO "HouseKeepers" VALUES(3,'alice.brown@example.com','2021-03-01',40000.0,3,'Alice Brown');
INSERT INTO "HouseKeepers" VALUES(4,'mike.johnson@example.com','2021-04-01',36000.0,4,'Mike Johnson');
INSERT INTO "HouseKeepers" VALUES(5,'emma.williams@example.com','2021-05-01',39000.0,5,'Emma Williams');
CREATE TABLE LaundryAttendants(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    hireDate DATE,
    salary REAL ,
    hotelId INTEGER, FullName TEXT,
    FOREIGN KEY(hotelId) REFERENCES Hotels(id)
);
INSERT INTO "LaundryAttendants" VALUES(1,'david.martin@example.com','2021-01-01',32000.0,1,'David Martin');
INSERT INTO "LaundryAttendants" VALUES(2,'lucy.taylor@example.com','2021-02-01',33000.0,2,'Lucy Taylor');
INSERT INTO "LaundryAttendants" VALUES(3,'charlie.thompson@example.com','2021-03-01',34000.0,3,'Charlie Thompson');
INSERT INTO "LaundryAttendants" VALUES(4,'grace.miller@example.com','2021-04-01',35000.0,4,'Grace Miller');
INSERT INTO "LaundryAttendants" VALUES(5,'samantha.wright@example.com','2021-05-01',36000.0,5,'Samantha Wright');
CREATE TABLE Rooms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    number INTEGER,
    hotelId INTEGER,
    houseKeeperId INTEGER,
    laundryAttendantId INTEGER,
    FOREIGN KEY(hotelId) REFERENCES Hotels(id),
    FOREIGN KEY(houseKeeperId) REFERENCES HouseKeepers(id),
    FOREIGN KEY(laundryAttendantId) REFERENCES LaundryAttendants(id)
);
INSERT INTO "Rooms" VALUES(1,101,1,1,1);
INSERT INTO "Rooms" VALUES(2,201,2,2,2);
INSERT INTO "Rooms" VALUES(3,301,3,3,3);
INSERT INTO "Rooms" VALUES(4,401,4,4,4);
INSERT INTO "Rooms" VALUES(5,501,5,5,5);
INSERT INTO "Rooms" VALUES(6,102,1,1,1);
INSERT INTO "Rooms" VALUES(7,202,2,2,2);
INSERT INTO "Rooms" VALUES(8,302,3,3,3);
INSERT INTO "Rooms" VALUES(9,402,4,4,4);
INSERT INTO "Rooms" VALUES(10,502,5,5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hotels',5);
INSERT INTO "sqlite_sequence" VALUES('HouseKeepers',5);
INSERT INTO "sqlite_sequence" VALUES('LaundryAttendants',5);
INSERT INTO "sqlite_sequence" VALUES('Rooms',10);
COMMIT;
