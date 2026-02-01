BEGIN TRANSACTION;
CREATE TABLE Ballrooms(
    id INTEGER PRIMARY KEY,
    name TEXT,
    seating_capacity INTEGER,
    price PERCENTAGE,
    availability BOOLEAN  DEFAULT 1
);
INSERT INTO "Ballrooms" VALUES(1,'Grand Hall',300,8,1);
INSERT INTO "Ballrooms" VALUES(2,'Conference Room',100,6,1);
INSERT INTO "Ballrooms" VALUES(3,'Executive Suite',50,10,1);
INSERT INTO "Ballrooms" VALUES(4,'Board Room',20,12,1);
INSERT INTO "Ballrooms" VALUES(5,'Meeting Room A',75,9,1);
CREATE TABLE Bookings(
    booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_id INTEGER,
    event_date TIMESTAMP,
    special_requests TEXT,
    FOREIGN KEY(guest_id) REFERENCES Guests(guest_id)
);
INSERT INTO "Bookings" VALUES(1,1,'2023-01-02 18:00:00','Vegan meal option');
INSERT INTO "Bookings" VALUES(2,2,'2023-02-16 14:00:00','Gluten-free menu');
INSERT INTO "Bookings" VALUES(3,3,'2023-03-21 20:00:00','Extra chairs required');
INSERT INTO "Bookings" VALUES(4,4,'2023-04-02 19:00:00','Music system setup');
INSERT INTO "Bookings" VALUES(5,5,'2023-12-21 22:00:00','Extended bar hours');
CREATE TABLE Events(
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ballroom_id INTEGER,
    greenroom_id INTEGER,
    event_name TEXT,
    date DATETIME,
    FOREIGN KEY(ballroom_id) REFERENCES Ballrooms(id),
    FOREIGN KEY(greenroom_id) REFERENCES GreenRooms(id)
);
INSERT INTO "Events" VALUES(1,1,1,'Annual Conference','2023-01-01');
INSERT INTO "Events" VALUES(2,2,NULL,'Company Meeting','2023-02-15');
INSERT INTO "Events" VALUES(3,NULL,2,'Wedding Reception','2023-03-20');
INSERT INTO "Events" VALUES(4,3,NULL,'Product Launch','2023-04-01');
INSERT INTO "Events" VALUES(5,4,4,'Holiday Party','2023-12-20');
CREATE TABLE GreenRooms(
    id INTEGER PRIMARY KEY,
    name TEXT,
    seating_capacity INTEGER,
    price PERCENTAGE,
    availability BOOLEAN  DEFAULT 1
);
INSERT INTO "GreenRooms" VALUES(1,'Terrace Garden',200,7,1);
INSERT INTO "GreenRooms" VALUES(2,'Rooftop Lounge',150,9,1);
INSERT INTO "GreenRooms" VALUES(3,'Poolside Patio',100,8,1);
INSERT INTO "GreenRooms" VALUES(4,'Gazebo',50,10,1);
INSERT INTO "GreenRooms" VALUES(5,'Foyer',30,7,1);
CREATE TABLE Guests(
    guest_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT,
    event_id INTEGER,
    FOREIGN KEY(event_id) REFERENCES Events(event_id)
);
INSERT INTO "Guests" VALUES(1,'John','Doe','john.doe@example.com','+1234567890',1);
INSERT INTO "Guests" VALUES(2,'Jane','Smith','jane.smith@example.com','+0987654321',2);
INSERT INTO "Guests" VALUES(3,'Alice','Johnson','alice.johnson@example.com','+1122334455',3);
INSERT INTO "Guests" VALUES(4,'Bob','Williams','bob.williams@example.com','+2233445566',4);
INSERT INTO "Guests" VALUES(5,'Charlie','Miller','charlie.miller@example.com','+3344556677',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Events',5);
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
COMMIT;
