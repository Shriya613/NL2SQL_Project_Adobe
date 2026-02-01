BEGIN TRANSACTION;
CREATE TABLE Accommodations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accommodation_type TEXT,
    capacity INTEGER
);
INSERT INTO "Accommodations" VALUES(1,'Hotel',300);
INSERT INTO "Accommodations" VALUES(2,'Resort',600);
INSERT INTO "Accommodations" VALUES(3,'Hostel',80);
INSERT INTO "Accommodations" VALUES(4,'Bed & Breakfast',20);
INSERT INTO "Accommodations" VALUES(5,'Villa',10);
CREATE TABLE Bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    travel_agency_id INTEGER REFERENCES Travel_Agencies(id),
    destination_id INTEGER REFERENCES Destinations(id),
    accommodation_id INTEGER REFERENCES Accommodations(id),
    tour_id INTEGER REFERENCES Tours(id),
    date DATETIME,
    price DECIMAL(10,2)
);
INSERT INTO "Bookings" VALUES(1,1,1,1,1,'2023-06-15',3000);
INSERT INTO "Bookings" VALUES(2,1,2,2,2,'2023-06-20',2500);
INSERT INTO "Bookings" VALUES(3,2,1,3,3,'2023-07-01',4000);
INSERT INTO "Bookings" VALUES(4,2,2,4,4,'2023-07-06',3500);
INSERT INTO "Bookings" VALUES(5,3,1,5,5,'2023-07-11',4500);
CREATE TABLE Destinations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    destination_name TEXT
);
INSERT INTO "Destinations" VALUES(1,'Europe');
INSERT INTO "Destinations" VALUES(2,'Asia');
INSERT INTO "Destinations" VALUES(3,'Africa');
INSERT INTO "Destinations" VALUES(4,'North America');
INSERT INTO "Destinations" VALUES(5,'South America');
INSERT INTO "Destinations" VALUES(6,'Australia');
CREATE TABLE Tours (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tour_name TEXT,
    duration INTEGER
);
INSERT INTO "Tours" VALUES(1,'City Tour',1);
INSERT INTO "Tours" VALUES(2,'Beach Vacation',7);
INSERT INTO "Tours" VALUES(3,'Mountain Trekking',10);
INSERT INTO "Tours" VALUES(4,'Cultural Experience',14);
INSERT INTO "Tours" VALUES(5,'Adventure Package',12);
CREATE TABLE Travel_Agencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Travel_Agencies" VALUES(1,'WorldWide Travels','New York');
INSERT INTO "Travel_Agencies" VALUES(2,'EuroTrip Expeditions','London');
INSERT INTO "Travel_Agencies" VALUES(3,'Asian Adventures','Tokyo');
INSERT INTO "Travel_Agencies" VALUES(4,'African Explorers','Cape Town');
INSERT INTO "Travel_Agencies" VALUES(5,'Americas Discoveries','Miami');
CREATE TABLE Travel_Agencies_Destinations (
    travel_agency_id INTEGER REFERENCES Travel_Agencies(id),
    destination_id INTEGER REFERENCES Destinations(id),
    FOREIGN KEY (travel_agency_id, destination_id) REFERENCES Bookings(travel_agency_id, destination_id)
);
INSERT INTO "Travel_Agencies_Destinations" VALUES(1,1);
INSERT INTO "Travel_Agencies_Destinations" VALUES(1,2);
INSERT INTO "Travel_Agencies_Destinations" VALUES(2,1);
INSERT INTO "Travel_Agencies_Destinations" VALUES(2,2);
INSERT INTO "Travel_Agencies_Destinations" VALUES(3,1);
INSERT INTO "Travel_Agencies_Destinations" VALUES(4,1);
INSERT INTO "Travel_Agencies_Destinations" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Destinations',6);
INSERT INTO "sqlite_sequence" VALUES('Travel_Agencies',5);
INSERT INTO "sqlite_sequence" VALUES('Accommodations',5);
INSERT INTO "sqlite_sequence" VALUES('Tours',5);
INSERT INTO "sqlite_sequence" VALUES('Bookings',5);
COMMIT;
