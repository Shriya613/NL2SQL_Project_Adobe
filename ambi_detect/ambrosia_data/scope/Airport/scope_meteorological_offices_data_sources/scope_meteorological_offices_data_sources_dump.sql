BEGIN TRANSACTION;
CREATE TABLE Data_Sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT,
    description TEXT
);
INSERT INTO "Data_Sources" VALUES(1,'Automated_Weather_Station','Collects real-time weather data.');
INSERT INTO "Data_Sources" VALUES(2,'Radar_System','Tracks precipitation and storm systems.');
INSERT INTO "Data_Sources" VALUES(3,'Satellite','Monitors global atmospheric conditions.');
INSERT INTO "Data_Sources" VALUES(4,'Wind_Sensor','Measures wind speed and direction.');
INSERT INTO "Data_Sources" VALUES(5,'Temperature_Sensor','Records ambient temperature.');
CREATE TABLE Flights (
    flight_number INTEGER PRIMARY KEY AUTOINCREMENT,
    departure_time DATETIME,
    arrival_time DATETIME,
    destination_airport TEXT,
    airline_company TEXT
);
INSERT INTO "Flights" VALUES(1,'2022-12-01 10:00:00','2022-12-01 13:00:00','JFK','American Airlines');
INSERT INTO "Flights" VALUES(2,'2022-12-01 12:00:00','2022-12-01 16:00:00','ORD','United Airlines');
INSERT INTO "Flights" VALUES(3,'2022-12-01 14:00:00','2022-12-01 18:00:00','LAX','Delta Airlines');
INSERT INTO "Flights" VALUES(4,'2022-12-01 16:00:00','2022-12-01 20:00:00','DFW','Southwest Airlines');
INSERT INTO "Flights" VALUES(5,'2022-12-01 18:00:00','2022-12-01 22:00:00','ATL','Spirit Airlines');
CREATE TABLE Meteorological_Offices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Meteorological_Offices" VALUES(1,'Office A','New York');
INSERT INTO "Meteorological_Offices" VALUES(2,'Office B','Chicago');
INSERT INTO "Meteorological_Offices" VALUES(3,'Office C','Los Angeles');
INSERT INTO "Meteorological_Offices" VALUES(4,'Office D','Dallas');
INSERT INTO "Meteorological_Offices" VALUES(5,'Office E','Atlanta');
CREATE TABLE Meteorological_Offices_Data_Sources (
    meteo_office_id INTEGER,
    data_source_id INTEGER,
    FOREIGN KEY(meteo_office_id) REFERENCES Meteorological_Offices(id),
    FOREIGN KEY(data_source_id) REFERENCES Data_Sources(id)
);
INSERT INTO "Meteorological_Offices_Data_Sources" VALUES(1,1);
INSERT INTO "Meteorological_Offices_Data_Sources" VALUES(2,1);
INSERT INTO "Meteorological_Offices_Data_Sources" VALUES(3,1);
INSERT INTO "Meteorological_Offices_Data_Sources" VALUES(4,1);
INSERT INTO "Meteorological_Offices_Data_Sources" VALUES(5,1);
INSERT INTO "Meteorological_Offices_Data_Sources" VALUES(5,3);
CREATE TABLE Runways (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    length REAL,
    width REAL
);
CREATE TABLE Terminals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    terminal_name TEXT,
    capacity INTEGER
);
INSERT INTO "Terminals" VALUES(1,'Terminal 1',5000);
INSERT INTO "Terminals" VALUES(2,'Terminal 2',4500);
INSERT INTO "Terminals" VALUES(3,'Terminal 3',4000);
INSERT INTO "Terminals" VALUES(4,'Terminal 4',3500);
INSERT INTO "Terminals" VALUES(5,'Terminal 5',3000);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Meteorological_Offices',5);
INSERT INTO "sqlite_sequence" VALUES('Data_Sources',5);
INSERT INTO "sqlite_sequence" VALUES('Terminals',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
COMMIT;
