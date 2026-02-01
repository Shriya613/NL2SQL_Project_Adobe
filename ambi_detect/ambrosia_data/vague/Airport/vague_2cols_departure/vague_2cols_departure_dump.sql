BEGIN TRANSACTION;
CREATE TABLE airlines (
    airline_id INTEGER PRIMARY KEY,
    name TEXT,
    code TEXT UNIQUE
);
INSERT INTO "airlines" VALUES(1,'United','UA');
INSERT INTO "airlines" VALUES(2,'Delta','DL');
INSERT INTO "airlines" VALUES(3,'Southwest','WN');
INSERT INTO "airlines" VALUES(4,'American','AA');
INSERT INTO "airlines" VALUES(5,'Alaska','AS');
CREATE TABLE boarding_areas (
    boarding_area_id INTEGER PRIMARY KEY,
    terminal_number INTEGER,
    "gate" INTEGER ,
    FOREIGN KEY(terminal_number) REFERENCES terminals(terminal_number)
);
INSERT INTO "boarding_areas" VALUES(1,1,10);
INSERT INTO "boarding_areas" VALUES(2,1,12);
INSERT INTO "boarding_areas" VALUES(3,2,14);
INSERT INTO "boarding_areas" VALUES(4,2,16);
INSERT INTO "boarding_areas" VALUES(5,3,18);
CREATE TABLE flights (
    flight_id INTEGER PRIMARY KEY,
    departure_time DATETIME,
    arrival_time DATETIME,
    distance REAL,
    origin_airport_code TEXT,
    destination_airport_code TEXT,
    airline_id INTEGER,
    FOREIGN KEY(airline_id) REFERENCES airlines(airline_id)
);
INSERT INTO "flights" VALUES(1,'2022-10-01 08:00:00','2022-10-01 10:00:00',2500.0,'ORD','LAX',1);
INSERT INTO "flights" VALUES(2,'2022-10-01 12:00:00','2022-10-01 14:00:00',3000.0,'JFK','SEA',2);
INSERT INTO "flights" VALUES(3,'2022-10-01 16:00:00','2022-10-01 18:00:00',2700.0,'DFW','PHX',3);
INSERT INTO "flights" VALUES(4,'2022-10-01 19:00:00','2022-10-01 21:00:00',2200.0,'MIA','ATL',4);
INSERT INTO "flights" VALUES(5,'2022-10-01 22:00:00','2022-10-02 02:00:00',4500.0,'LAX','HNL',5);
CREATE TABLE flights_boarding_areas (
    flight_id INTEGER,
    boarding_area_id INTEGER,
    FOREIGN KEY(flight_id) REFERENCES flights(flight_id),
    FOREIGN KEY(boarding_area_id) REFERENCES boarding_areas(boarding_area_id),
    PRIMARY KEY(flight_id, boarding_area_id)
);
INSERT INTO "flights_boarding_areas" VALUES(1,1);
INSERT INTO "flights_boarding_areas" VALUES(2,3);
INSERT INTO "flights_boarding_areas" VALUES(3,2);
INSERT INTO "flights_boarding_areas" VALUES(4,4);
INSERT INTO "flights_boarding_areas" VALUES(5,5);
CREATE TABLE terminals (
    terminal_number INTEGER PRIMARY KEY,
    capacity INTEGER ,
    location TEXT
);
INSERT INTO "terminals" VALUES(1,50,'Terminal A');
INSERT INTO "terminals" VALUES(2,60,'Terminal B');
INSERT INTO "terminals" VALUES(3,70,'Terminal C');
INSERT INTO "terminals" VALUES(4,45,'Terminal D');
INSERT INTO "terminals" VALUES(5,30,'Terminal E');
COMMIT;
