BEGIN TRANSACTION;
CREATE TABLE Aircraft (
    aircraft_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    capacity INTEGER
);
INSERT INTO "Aircraft" VALUES(1,'Airbus A380',450);
INSERT INTO "Aircraft" VALUES(2,'Boeing 747',400);
INSERT INTO "Aircraft" VALUES(3,'Airbus A321',200);
INSERT INTO "Aircraft" VALUES(4,'Boeing 737',180);
INSERT INTO "Aircraft" VALUES(5,'Embraer E190',110);
CREATE TABLE Facility (
    facility_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Facility" VALUES(1,'Baggage Claim','A place where passengers collect their luggage');
INSERT INTO "Facility" VALUES(2,'Restaurant','A place where passengers can eat');
INSERT INTO "Facility" VALUES(3,'Shopping Center','A place where passengers can shop');
INSERT INTO "Facility" VALUES(4,'Information Desk','A place where passengers can get information about flights');
INSERT INTO "Facility" VALUES(5,'Currency Exchange','A place where passengers can exchange currency');
CREATE TABLE Flight (
    flight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    airline_name TEXT,
    departure_time TIMESTAMP,
    arrival_time TIMESTAMP
);
INSERT INTO "Flight" VALUES(1,'United Airlines','08-05-2022 9:00 AM','08-05-2022 11:00 PM');
INSERT INTO "Flight" VALUES(2,'Delta Airlines','08-05-2022 10:00 AM','08-05-2022 12:00 PM');
INSERT INTO "Flight" VALUES(3,'American Airlines','08-05-2022 11:00 AM','08-05-2022 1:00 PM');
INSERT INTO "Flight" VALUES(4,'Southwest Airlines','08-05-2022 12:00 PM','08-05-2022 2:00 PM');
INSERT INTO "Flight" VALUES(5,'Spirit Airlines','08-05-2022 1:00 PM','08-05-2022 3:00 PM');
CREATE TABLE Gate (
    gate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    airport_code TEXT,
    terminal_number INTEGER
);
INSERT INTO "Gate" VALUES(1,'JFK',3);
INSERT INTO "Gate" VALUES(2,'LAX',2);
INSERT INTO "Gate" VALUES(3,'ORD',1);
INSERT INTO "Gate" VALUES(4,'ATL',6);
INSERT INTO "Gate" VALUES(5,'YUL',7);
CREATE TABLE Gate_Facility (
    gate_facility_id INTEGER PRIMARY KEY AUTOINCREMENT,
    gate_id INTEGER,
    facility_id INTEGER,
    FOREIGN KEY(gate_id) REFERENCES Gate(gate_id),
    FOREIGN KEY(facility_id) REFERENCES Facility(facility_id)
);
INSERT INTO "Gate_Facility" VALUES(1,1,1);
INSERT INTO "Gate_Facility" VALUES(2,1,2);
INSERT INTO "Gate_Facility" VALUES(3,2,3);
INSERT INTO "Gate_Facility" VALUES(4,3,4);
INSERT INTO "Gate_Facility" VALUES(5,4,5);
INSERT INTO "Gate_Facility" VALUES(6,2,1);
INSERT INTO "Gate_Facility" VALUES(7,3,1);
INSERT INTO "Gate_Facility" VALUES(8,4,1);
INSERT INTO "Gate_Facility" VALUES(9,5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Gate',5);
INSERT INTO "sqlite_sequence" VALUES('Facility',5);
INSERT INTO "sqlite_sequence" VALUES('Gate_Facility',9);
INSERT INTO "sqlite_sequence" VALUES('Flight',5);
INSERT INTO "sqlite_sequence" VALUES('Aircraft',5);
COMMIT;
