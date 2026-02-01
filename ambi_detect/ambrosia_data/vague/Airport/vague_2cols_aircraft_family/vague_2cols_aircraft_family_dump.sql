BEGIN TRANSACTION;
CREATE TABLE Airlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airline_name TEXT,
    headquarters_city TEXT,
    headquarters_country TEXT
);
INSERT INTO "Airlines" VALUES(1,'American Airlines','Fort Worth','USA');
INSERT INTO "Airlines" VALUES(2,'Delta Air Lines','Atlanta','USA');
INSERT INTO "Airlines" VALUES(3,'Emirates','Dubai','UAE');
INSERT INTO "Airlines" VALUES(4,'British Airways','London','UK');
INSERT INTO "Airlines" VALUES(5,'Air France','Paris','France');
CREATE TABLE Airports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city TEXT,
    country TEXT
);
INSERT INTO "Airports" VALUES(1,'John F. Kennedy International','New York','USA');
INSERT INTO "Airports" VALUES(2,'Los Angeles International','Los Angeles','USA');
INSERT INTO "Airports" VALUES(3,'Dubai International','Dubai','UAE');
INSERT INTO "Airports" VALUES(4,'Heathrow Airport','London','UK');
INSERT INTO "Airports" VALUES(5,'Charles de Gaulle Airport','Paris','France');
CREATE TABLE AssignsPilotToFlight (
    pilot_id INTEGER,
    flight_id INTEGER,
    assignment_date DATE,
    FOREIGN KEY (pilot_id) REFERENCES Pilots (id),
    FOREIGN KEY (flight_id) REFERENCES Flights (id)
);
INSERT INTO "AssignsPilotToFlight" VALUES(1,1,'2022-01-01');
INSERT INTO "AssignsPilotToFlight" VALUES(2,2,'2022-01-01');
INSERT INTO "AssignsPilotToFlight" VALUES(3,3,'2022-01-01');
INSERT INTO "AssignsPilotToFlight" VALUES(4,4,'2022-01-01');
INSERT INTO "AssignsPilotToFlight" VALUES(5,5,'2022-01-01');
CREATE TABLE Flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id INTEGER,
    plane_id INTEGER,
    flight_number TEXT,
    departure_date DATE,
    scheduled_departure_time TIME,
    actual_departure_time TIME,
    FOREIGN KEY (route_id) REFERENCES Routes (id),
    FOREIGN KEY (plane_id) REFERENCES Planes (id)
);
INSERT INTO "Flights" VALUES(1,1,1,'AA123','2022-01-01','12:00:00',NULL);
INSERT INTO "Flights" VALUES(2,2,2,'EK123','2022-01-01','14:00:00',NULL);
INSERT INTO "Flights" VALUES(3,NULL,2,'BA123','2022-01-01','10:00:00',NULL);
INSERT INTO "Flights" VALUES(4,4,2,'AF123','2022-01-01','16:00:00',NULL);
INSERT INTO "Flights" VALUES(5,5,NULL,'DL123','2022-01-01','15:00:00',NULL);
CREATE TABLE Pilots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license_number TEXT,
    rating_level TEXT,
    total_flight_hours DECIMAL
, full_name TEXT);
INSERT INTO "Pilots" VALUES(1,'ABCD1234','ATPL',5000,'John Doe');
INSERT INTO "Pilots" VALUES(2,'EFGH5678','ATPL',7000,'Jane Smith');
INSERT INTO "Pilots" VALUES(3,'IJKL9012','ATPL',6000,'Michael Johnson');
INSERT INTO "Pilots" VALUES(4,'LMNO3456','ATPL',4500,'Sarah Williams');
INSERT INTO "Pilots" VALUES(5,'PQRS7890','ATPL',5500,'David Brown');
CREATE TABLE Planes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer_name TEXT,
    model_name TEXT,
    "purpose" TEXT,
    max_passenger_capacity INT ,
    range_km REAL
);
INSERT INTO "Planes" VALUES(1,'Boeing','787-9 Dreamliner','Passenger',280,13600.0);
INSERT INTO "Planes" VALUES(2,'Airbus','A350-1000','Passenger',440,14800.0);
INSERT INTO "Planes" VALUES(3,'Bombardier Aerospace','Global 7500','Business Jet',19,7700.0);
INSERT INTO "Planes" VALUES(4,'Gulfstream Aerospace','G650ER','Business Jet',18,7500.0);
INSERT INTO "Planes" VALUES(5,'Embraer Executive Jets','Legacy 650','Business Jet',12,6000.0);
CREATE TABLE Routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_airport_id INTEGER,
    destination_airport_id INTEGER,
    airline_id INTEGER,
    departure_time TIMESTAMP,
    arrival_time TIMESTAMP,
    FOREIGN KEY (source_airport_id) REFERENCES Airports (id),
    FOREIGN KEY (destination_airport_id) REFERENCES Airports (id),
    FOREIGN KEY (airline_id) REFERENCES Airlines (id)
);
INSERT INTO "Routes" VALUES(1,1,2,1,'2022-01-01 12:00:00','2022-01-01 16:00:00');
INSERT INTO "Routes" VALUES(2,2,3,3,'2022-01-01 14:00:00','2022-01-01 20:00:00');
INSERT INTO "Routes" VALUES(3,4,NULL,4,'2022-01-01 10:00:00','2022-01-01 14:00:00');
INSERT INTO "Routes" VALUES(4,5,1,5,'2022-01-01 16:00:00','2022-01-01 10:00:00');
INSERT INTO "Routes" VALUES(5,1,2,2,'2022-01-01 15:00:00','2022-01-01 19:00:00');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Planes',5);
INSERT INTO "sqlite_sequence" VALUES('Routes',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Pilots',5);
COMMIT;
