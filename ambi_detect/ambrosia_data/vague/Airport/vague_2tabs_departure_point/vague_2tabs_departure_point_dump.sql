BEGIN TRANSACTION;
CREATE TABLE Airline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airline_name TEXT,
    headquarter_city_id INTEGER,
    FOREIGN KEY(headquarter_city_id) REFERENCES City(id)
);
INSERT INTO "Airline" VALUES(1,'American Airlines',1);
INSERT INTO "Airline" VALUES(2,'British Airways',2);
INSERT INTO "Airline" VALUES(3,'All Nippon Airways',3);
INSERT INTO "Airline" VALUES(4,'Air France',4);
INSERT INTO "Airline" VALUES(5,'Emirates',5);
CREATE TABLE City (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    country TEXT
);
INSERT INTO "City" VALUES(1,'New York','USA');
INSERT INTO "City" VALUES(2,'London','UK');
INSERT INTO "City" VALUES(3,'Tokyo','Japan');
INSERT INTO "City" VALUES(4,'Paris','France');
INSERT INTO "City" VALUES(5,'Dubai','UAE');
CREATE TABLE Flight (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number TEXT,
    departure_time TIMESTAMP,
    arrival_time TIMESTAMP,
    terminal_id INTEGER,
    origin_city_id INTEGER,
    destination_city_id INTEGER,
    FOREIGN KEY(terminal_id) REFERENCES Terminal(id),
    FOREIGN KEY(origin_city_id) REFERENCES City(id),
    FOREIGN KEY(destination_city_id) REFERENCES City(id)
);
INSERT INTO "Flight" VALUES(1,'AA101','2022-12-01 08:00:00','2022-12-01 10:00:00',1,1,3);
INSERT INTO "Flight" VALUES(2,'BA201','2022-12-01 11:00:00','2022-12-01 13:00:00',2,2,4);
INSERT INTO "Flight" VALUES(3,'JL301','2022-12-01 15:00:00','2022-12-01 17:00:00',3,3,5);
INSERT INTO "Flight" VALUES(4,'AF401','2022-12-01 19:00:00','2022-12-01 21:00:00',4,4,1);
INSERT INTO "Flight" VALUES(5,'EK601','2022-12-01 22:00:00','2022-12-02 02:00:00',5,5,2);
CREATE TABLE Pilot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pilot_license_number TEXT,
    home_airport_code TEXT,
    FOREIGN KEY(home_airport_code) REFERENCES City(name)
);
INSERT INTO "Pilot" VALUES(1,'ABC1234','NY');
INSERT INTO "Pilot" VALUES(2,'DEF5678','LDN');
INSERT INTO "Pilot" VALUES(3,'GHI9012','TYO');
INSERT INTO "Pilot" VALUES(4,'JKL3456','CDG');
INSERT INTO "Pilot" VALUES(5,'MNO7890','DBX');
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
CREATE TABLE "Terminal" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    terminal_name TEXT
);
INSERT INTO "Terminal" VALUES(1,'Terminal 1');
INSERT INTO "Terminal" VALUES(2,'Terminal 2');
INSERT INTO "Terminal" VALUES(3,'Terminal 3');
INSERT INTO "Terminal" VALUES(4,'Terminal 4');
INSERT INTO "Terminal" VALUES(5,'Terminal 5');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('City',5);
INSERT INTO "sqlite_sequence" VALUES('Flight',5);
INSERT INTO "sqlite_sequence" VALUES('Airline',5);
INSERT INTO "sqlite_sequence" VALUES('Pilot',5);
INSERT INTO "sqlite_sequence" VALUES('Plane',5);
INSERT INTO "sqlite_sequence" VALUES('Terminal',5);
COMMIT;
