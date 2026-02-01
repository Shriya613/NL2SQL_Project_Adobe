BEGIN TRANSACTION;
CREATE TABLE "Airline" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airline_name TEXT);
INSERT INTO "Airline" VALUES(1,'American Airlines');
INSERT INTO "Airline" VALUES(2,'British Airways');
INSERT INTO "Airline" VALUES(3,'All Nippon Airways');
INSERT INTO "Airline" VALUES(4,'Air France');
INSERT INTO "Airline" VALUES(5,'Emirates');
CREATE TABLE "Departure_Point" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    terminal_name TEXT
);
INSERT INTO "Departure_Point" VALUES(1,'Terminal 1');
INSERT INTO "Departure_Point" VALUES(2,'Terminal 2');
INSERT INTO "Departure_Point" VALUES(3,'Terminal 3');
INSERT INTO "Departure_Point" VALUES(4,'Terminal 4');
INSERT INTO "Departure_Point" VALUES(5,'Terminal 5');
CREATE TABLE "Flight" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number TEXT,
    departure_time TIMESTAMP,
    arrival_time TIMESTAMP,
    terminal_id INTEGER,
    origin_city_id INTEGER,
    FOREIGN KEY(terminal_id) REFERENCES "Departure_Point"(id),
    FOREIGN KEY(origin_city_id) REFERENCES City(id));
INSERT INTO "Flight" VALUES(1,'AA101','2022-12-01 08:00:00','2022-12-01 10:00:00',1,1);
INSERT INTO "Flight" VALUES(2,'BA201','2022-12-01 11:00:00','2022-12-01 13:00:00',2,2);
INSERT INTO "Flight" VALUES(3,'JL301','2022-12-01 15:00:00','2022-12-01 17:00:00',3,3);
INSERT INTO "Flight" VALUES(4,'AF401','2022-12-01 19:00:00','2022-12-01 21:00:00',4,4);
INSERT INTO "Flight" VALUES(5,'EK601','2022-12-01 22:00:00','2022-12-02 02:00:00',5,5);
CREATE TABLE "Pilot" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pilot_license_number TEXT);
INSERT INTO "Pilot" VALUES(1,'ABC1234');
INSERT INTO "Pilot" VALUES(2,'DEF5678');
INSERT INTO "Pilot" VALUES(3,'GHI9012');
INSERT INTO "Pilot" VALUES(4,'JKL3456');
INSERT INTO "Pilot" VALUES(5,'MNO7890');
CREATE TABLE Plane (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plane_type TEXT,
    capacity INTEGER,
    registered_to_airline_id INTEGER,
    current_pilot_id INTEGER,
    FOREIGN KEY(registered_to_airline_id) REFERENCES Airline(id),
    FOREIGN KEY(current_pilot_id) REFERENCES Pilot(id)
);
INSERT INTO "Plane" VALUES(1,'Boeing 777',350,1,1);
INSERT INTO "Plane" VALUES(2,'Boeing 747',400,2,2);
INSERT INTO "Plane" VALUES(3,'Airbus A380',500,3,3);
INSERT INTO "Plane" VALUES(4,'Airbus A320',200,4,4);
INSERT INTO "Plane" VALUES(5,'Bombardier CRJ1000',90,5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Plane',5);
INSERT INTO "sqlite_sequence" VALUES('Departure_Point',5);
INSERT INTO "sqlite_sequence" VALUES('Flight',5);
INSERT INTO "sqlite_sequence" VALUES('Airline',5);
INSERT INTO "sqlite_sequence" VALUES('Pilot',5);
COMMIT;
