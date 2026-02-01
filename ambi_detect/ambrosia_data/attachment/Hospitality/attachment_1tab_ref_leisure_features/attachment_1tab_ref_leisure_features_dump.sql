BEGIN TRANSACTION;
CREATE TABLE City(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "City" VALUES(1,'New York');
INSERT INTO "City" VALUES(2,'Los Angeles');
INSERT INTO "City" VALUES(3,'Chicago');
INSERT INTO "City" VALUES(4,'Houston');
INSERT INTO "City" VALUES(5,'Phoenix');
CREATE TABLE FeatureInRoom(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roomtype_id INTEGER,
    leisurefeature_id INTEGER,
    FOREIGN KEY(roomtype_id) REFERENCES RoomType(id),
    FOREIGN KEY(leisurefeature_id) REFERENCES LeisureFeature(id)
);
INSERT INTO "FeatureInRoom" VALUES(1,1,1);
INSERT INTO "FeatureInRoom" VALUES(2,2,2);
INSERT INTO "FeatureInRoom" VALUES(3,3,1);
INSERT INTO "FeatureInRoom" VALUES(4,4,2);
INSERT INTO "FeatureInRoom" VALUES(5,5,1);
CREATE TABLE Hotel(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    city_id INTEGER,
    FOREIGN KEY(city_id) REFERENCES City(id)
);
INSERT INTO "Hotel" VALUES(1,'The Plaza','Central Park South, New York, NY 10019',1);
INSERT INTO "Hotel" VALUES(2,'Beverly Wilshire','9500 Wilshire Blvd., Beverly Hills, CA 90212',2);
INSERT INTO "Hotel" VALUES(3,'Palmer House Hilton','17 E Monroe St., Chicago, IL 60603',3);
INSERT INTO "Hotel" VALUES(4,'Marriott Marquis Houston','1777 Walker St., Houston, TX 77010',4);
INSERT INTO "Hotel" VALUES(5,'Fairmont Scottsdale Princess','7575 East Princess Dr., Phoenix, AZ 85054',5);
CREATE TABLE LeisureFeature(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature TEXT UNIQUE
);
INSERT INTO "LeisureFeature" VALUES(1,'Swimming Pool');
INSERT INTO "LeisureFeature" VALUES(2,'Hot Tub');
CREATE TABLE Reservation(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkin_date DATE,
    checkout_date DATE,
    roomtype_id INTEGER,
    customer_name TEXT,
    guest_count INT ,
    FOREIGN KEY(roomtype_id) REFERENCES RoomType(id)
);
INSERT INTO "Reservation" VALUES(1,'2023-01-01','2023-01-05',1,'John Doe',2);
INSERT INTO "Reservation" VALUES(2,'2023-01-10','2023-01-15',2,'Jane Smith',3);
INSERT INTO "Reservation" VALUES(3,'2023-02-01','2023-02-05',3,'Alice Johnson',1);
INSERT INTO "Reservation" VALUES(4,'2023-02-10','2023-02-15',4,'Bob Brown',2);
INSERT INTO "Reservation" VALUES(5,'2023-03-01','2023-03-05',5,'Carol White',1);
CREATE TABLE RoomType(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hotel_id INTEGER,
    type TEXT,
    price DECIMAL(10,2),
    FOREIGN KEY(hotel_id) REFERENCES Hotel(id)
);
INSERT INTO "RoomType" VALUES(1,1,'Standard',250);
INSERT INTO "RoomType" VALUES(2,1,'Deluxe',300);
INSERT INTO "RoomType" VALUES(3,2,'Luxury Suite',700);
INSERT INTO "RoomType" VALUES(4,3,'Executive Suite',400);
INSERT INTO "RoomType" VALUES(5,4,'Family Suite',250);
INSERT INTO "RoomType" VALUES(6,5,'Garden View',200);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('City',5);
INSERT INTO "sqlite_sequence" VALUES('Hotel',5);
INSERT INTO "sqlite_sequence" VALUES('RoomType',6);
INSERT INTO "sqlite_sequence" VALUES('LeisureFeature',2);
INSERT INTO "sqlite_sequence" VALUES('FeatureInRoom',5);
INSERT INTO "sqlite_sequence" VALUES('Reservation',5);
COMMIT;
