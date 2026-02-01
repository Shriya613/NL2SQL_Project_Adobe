BEGIN TRANSACTION;
CREATE TABLE Airlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    headquarter_city TEXT,
    headquarter_country TEXT
);
INSERT INTO "Airlines" VALUES(1,'Airline1','HeadquartersA','HQX');
INSERT INTO "Airlines" VALUES(2,'Airline2','HeadquartersB','HQY');
INSERT INTO "Airlines" VALUES(3,'Airline3','HeadquartersC','HQZ');
INSERT INTO "Airlines" VALUES(4,'Airline4','HeadquartersD','HQW');
INSERT INTO "Airlines" VALUES(5,'Airline5','HeadquartersE','HQV');
CREATE TABLE Airports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city TEXT,
    country TEXT
);
INSERT INTO "Airports" VALUES(1,'Airport1','CityA','CountryX');
INSERT INTO "Airports" VALUES(2,'Airport2','CityB','CountryY');
INSERT INTO "Airports" VALUES(3,'Airport3','CityC','CountryZ');
INSERT INTO "Airports" VALUES(4,'Airport4','CityD','CountryW');
INSERT INTO "Airports" VALUES(5,'Airport5','CityE','CountryV');
CREATE TABLE Flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    departure_time TIMESTAMP,
    arrival_time TIMESTAMP,
    runway_id INTEGER,
    plane_model TEXT,
    FOREIGN KEY(runway_id) REFERENCES Runways(id),
    FOREIGN KEY(plane_model) REFERENCES Planes(model)
);
INSERT INTO "Flights" VALUES(1,'2022-01-01 08:00:00','2022-01-01 10:00:00',1,'PlaneModel1');
INSERT INTO "Flights" VALUES(2,'2022-01-02 09:00:00','2022-01-02 11:00:00',2,'PlaneModel2');
INSERT INTO "Flights" VALUES(3,'2022-01-03 10:00:00','2022-01-03 12:00:00',3,'PlaneModel3');
INSERT INTO "Flights" VALUES(4,'2022-01-04 11:00:00','2022-01-04 13:00:00',4,'PlaneModel4');
INSERT INTO "Flights" VALUES(5,'2022-01-05 12:00:00','2022-01-05 14:00:00',5,'PlaneModel5');
CREATE TABLE Passengers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fullname TEXT,
    passport TEXT,
    flight_id INTEGER,
    FOREIGN KEY(flight_id) REFERENCES Flights(id)
);
INSERT INTO "Passengers" VALUES(1,'Passenger1','Passport1',1);
INSERT INTO "Passengers" VALUES(2,'Passenger2','Passport2',2);
INSERT INTO "Passengers" VALUES(3,'Passenger3','Passport3',3);
INSERT INTO "Passengers" VALUES(4,'Passenger4','Passport4',4);
INSERT INTO "Passengers" VALUES(5,'Passenger5','Passport5',5);
CREATE TABLE Planes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    linear_dimensions TEXT,
    weight_restriction REAL,
    airline_id INTEGER,
    FOREIGN KEY(airline_id) REFERENCES Airlines(id)
);
INSERT INTO "Planes" VALUES(1,'PlaneModel1','Dimensions1',10000.0,1);
INSERT INTO "Planes" VALUES(2,'PlaneModel2','Dimensions2',20000.0,2);
INSERT INTO "Planes" VALUES(3,'PlaneModel3','Dimensions3',30000.0,3);
INSERT INTO "Planes" VALUES(4,'PlaneModel4','Dimensions4',40000.0,4);
INSERT INTO "Planes" VALUES(5,'PlaneModel5','Dimensions5',50000.0,5);
CREATE TABLE Runways (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    length REAL,
    width REAL,
    capacity INTEGER,
    airport_id INTEGER,
    FOREIGN KEY(airport_id) REFERENCES Airports(id)
);
INSERT INTO "Runways" VALUES(1,8000.0,100.0,9999,1);
INSERT INTO "Runways" VALUES(2,7000.0,150.0,9998,2);
INSERT INTO "Runways" VALUES(3,6000.0,120.0,9997,3);
INSERT INTO "Runways" VALUES(4,5000.0,90.0,9996,4);
INSERT INTO "Runways" VALUES(5,4000.0,80.0,9995,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Planes',5);
INSERT INTO "sqlite_sequence" VALUES('Runways',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Passengers',5);
COMMIT;
