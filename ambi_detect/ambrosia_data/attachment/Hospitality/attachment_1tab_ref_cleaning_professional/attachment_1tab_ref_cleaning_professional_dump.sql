BEGIN TRANSACTION;
CREATE TABLE CleaningProfessionals (
    id INTEGER PRIMARY KEY,
    name TEXT,
    type TEXT ,
    hotel_id INTEGER,
    FOREIGN KEY (hotel_id) REFERENCES Hotels(id)
);
INSERT INTO "CleaningProfessionals" VALUES(1,'Alice Johnson','Housekeeper',1);
INSERT INTO "CleaningProfessionals" VALUES(2,'Bob Brown','Housekeeper',2);
INSERT INTO "CleaningProfessionals" VALUES(3,'Charlie White','Laundry Attendant',1);
INSERT INTO "CleaningProfessionals" VALUES(4,'David Green','Laundry Attendant',2);
INSERT INTO "CleaningProfessionals" VALUES(5,'Eva Blue','Housekeeper',1);
INSERT INTO "CleaningProfessionals" VALUES(6,'Frank Red','Laundry Attendant',2);
CREATE TABLE Customers (
    id INTEGER PRIMARY KEY,
    full_name TEXT,
    email TEXT,
    phone TEXT
);
INSERT INTO "Customers" VALUES(1,'John Doe','john@example.com','123-456-7890');
INSERT INTO "Customers" VALUES(2,'Jane Smith','jane@example.com','098-765-4321');
CREATE TABLE Hotels (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Hotels" VALUES(1,'The Grand Hotel');
INSERT INTO "Hotels" VALUES(2,'Park View Inn');
CREATE TABLE Reservations (
    id INTEGER PRIMARY KEY,
    checkin DATE,
    checkout DATE,
    room_id INTEGER,
    customer_name TEXT,
    FOREIGN KEY (room_id) REFERENCES Rooms(id)
);
INSERT INTO "Reservations" VALUES(1,'2023-01-01','2023-01-07',101,'John Doe');
INSERT INTO "Reservations" VALUES(2,'2023-01-14','2023-01-21',201,'Jane Smith');
CREATE TABLE Rooms (
    id INTEGER PRIMARY KEY,
    room_number INTEGER,
    hotel_id INTEGER,
    FOREIGN KEY (hotel_id) REFERENCES Hotels(id)
);
INSERT INTO "Rooms" VALUES(1,101,1);
INSERT INTO "Rooms" VALUES(2,201,1);
INSERT INTO "Rooms" VALUES(3,301,2);
INSERT INTO "Rooms" VALUES(4,401,2);
INSERT INTO "Rooms" VALUES(5,501,2);
COMMIT;
