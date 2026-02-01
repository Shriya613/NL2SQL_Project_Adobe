BEGIN TRANSACTION;
CREATE TABLE Airport (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airport_code TEXT,
    name TEXT,
    city TEXT,
    country TEXT
);
INSERT INTO "Airport" VALUES(1,'LAX','Los Angeles International Airport','Los Angeles','USA');
INSERT INTO "Airport" VALUES(2,'CDG','Charles de Gaulle Airport','Paris','France');
INSERT INTO "Airport" VALUES(3,'GRU','Guarulhos–Governador André Franco Montoro International Airport','São Paulo','Brazil');
INSERT INTO "Airport" VALUES(4,'PEK','Capital International Airport','Beijing','China');
INSERT INTO "Airport" VALUES(5,'PVG','Shanghai Pudong International Airport','Shanghai','China');
CREATE TABLE Flight (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_date DATETIME,
    departure_airport_id INTEGER REFERENCES Airport(id),
    arrival_airport_id INTEGER REFERENCES Airport(id),
    plane_id INTEGER REFERENCES Plane(id)
);
INSERT INTO "Flight" VALUES(1,'2022-01-01 08:00:00',1,2,1);
INSERT INTO "Flight" VALUES(2,'2022-01-01 12:00:00',2,3,2);
INSERT INTO "Flight" VALUES(3,'2022-01-01 16:00:00',3,4,3);
INSERT INTO "Flight" VALUES(4,'2022-01-01 20:00:00',4,5,4);
INSERT INTO "Flight" VALUES(5,'2022-01-01 22:00:00',5,1,5);
CREATE TABLE Manufacturer_Name (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer_name TEXT,
    country TEXT
);
INSERT INTO "Manufacturer_Name" VALUES(1,'Boeing','USA');
INSERT INTO "Manufacturer_Name" VALUES(2,'Airbus','France');
INSERT INTO "Manufacturer_Name" VALUES(3,'Embraer','Brazil');
INSERT INTO "Manufacturer_Name" VALUES(4,'Bombardier','Canada');
INSERT INTO "Manufacturer_Name" VALUES(5,'Comac','China');
CREATE TABLE Model_Name (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    year_introduced INTEGER,
    MAX_PASSENGERS INT
);
INSERT INTO "Model_Name" VALUES(1,'737 Max',2017,220);
INSERT INTO "Model_Name" VALUES(2,'A320neo',2016,180);
INSERT INTO "Model_Name" VALUES(3,'ERJ-190',2007,100);
INSERT INTO "Model_Name" VALUES(4,'CRJ-900',2000,85);
INSERT INTO "Model_Name" VALUES(5,'ARJ-21',2009,90);
CREATE TABLE Plane (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plane_model_id INTEGER REFERENCES Model_Name(id),
    manufacturer_id INTEGER REFERENCES Manufacturer_Name(id),
    registration_number TEXT,
    aircraft_type TEXT,
    current_location TEXT
);
INSERT INTO "Plane" VALUES(1,1,1,'N123AB','Passenger','Los Angeles');
INSERT INTO "Plane" VALUES(2,2,2,'F-HSBN','Passenger','Paris');
INSERT INTO "Plane" VALUES(3,3,3,'PP-XXX','Regional Passenger','São Paulo');
INSERT INTO "Plane" VALUES(4,4,4,'B-HLDY','Regional Passenger','Beijing');
INSERT INTO "Plane" VALUES(5,5,5,'B-HLEU','Business Jet','Shanghai');
CREATE TABLE Runway (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    runway_length REAL,
    runway_width REAL,
    airport_id INTEGER REFERENCES Airport(id)
);
INSERT INTO "Runway" VALUES(1,10000.0,150.0,1);
INSERT INTO "Runway" VALUES(2,2800.0,45.0,2);
INSERT INTO "Runway" VALUES(3,2400.0,45.0,3);
INSERT INTO "Runway" VALUES(4,3000.0,45.0,4);
INSERT INTO "Runway" VALUES(5,2800.0,45.0,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Manufacturer_Name',5);
INSERT INTO "sqlite_sequence" VALUES('Model_Name',5);
INSERT INTO "sqlite_sequence" VALUES('Plane',5);
INSERT INTO "sqlite_sequence" VALUES('Airport',5);
INSERT INTO "sqlite_sequence" VALUES('Runway',5);
INSERT INTO "sqlite_sequence" VALUES('Flight',5);
COMMIT;
