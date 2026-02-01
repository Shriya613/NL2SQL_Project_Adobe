BEGIN TRANSACTION;
CREATE TABLE Airports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    city TEXT
);
INSERT INTO "Airports" VALUES(1,'JFK International','New York');
INSERT INTO "Airports" VALUES(2,'LAX International','Los Angeles');
INSERT INTO "Airports" VALUES(3,'ORD International','Chicago');
INSERT INTO "Airports" VALUES(4,'ATL Hartsfield-Jackson','Atlanta');
INSERT INTO "Airports" VALUES(5,'SEA Seattle/Tacoma','Seattle');
CREATE TABLE Flights(
    flightID INTEGER PRIMARY KEY AUTOINCREMENT,
    departureGateId INTEGER,
    arrivalGateId INTEGER,
    departureTime TIMESTAMP,
    arrivalTime TIMESTAMP,
    departureAirportId INTEGER,
    arrivalAirportId INTEGER,
    FOREIGN KEY(departureGateId) REFERENCES Gates(id),
    FOREIGN KEY(arrivalGateId) REFERENCES Gates(id),
    FOREIGN KEY(departureAirportId) REFERENCES Airports(id),
    FOREIGN KEY(arrivalAirportId) REFERENCES Airports(id)
);
INSERT INTO "Flights" VALUES(1,1,1,'2022-09-01 07:00:00','2022-09-01 08:00:00',1,2);
INSERT INTO "Flights" VALUES(2,2,2,'2022-09-01 08:00:00','2022-09-01 09:00:00',1,2);
INSERT INTO "Flights" VALUES(3,3,3,'2022-09-01 09:00:00','2022-09-01 10:00:00',1,2);
INSERT INTO "Flights" VALUES(4,4,4,'2022-09-01 10:00:00','2022-09-01 11:00:00',1,2);
INSERT INTO "Flights" VALUES(5,5,5,'2022-09-01 11:00:00','2022-09-01 12:00:00',1,2);
CREATE TABLE Gates(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    terminalId INTEGER,
    gateNumber INTEGER,
    FOREIGN KEY(terminalId) REFERENCES Terminals(id)
);
INSERT INTO "Gates" VALUES(1,1,10);
INSERT INTO "Gates" VALUES(2,1,20);
INSERT INTO "Gates" VALUES(3,1,30);
INSERT INTO "Gates" VALUES(4,2,10);
INSERT INTO "Gates" VALUES(5,2,20);
INSERT INTO "Gates" VALUES(6,3,10);
INSERT INTO "Gates" VALUES(7,3,20);
INSERT INTO "Gates" VALUES(8,4,10);
INSERT INTO "Gates" VALUES(9,4,20);
CREATE TABLE "RetailOutlets"(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
    type TEXT ,
    location TEXT,
    terminalId INTEGER,
    FOREIGN KEY(terminalId) REFERENCES Terminals(id)
);
INSERT INTO "RetailOutlets" VALUES(1,'Outlet 1','Duty-Free Shop','Terminal 3',3);
INSERT INTO "RetailOutlets" VALUES(2,'Outlet 2','Duty-Free Shop','Concourse E',2);
INSERT INTO "RetailOutlets" VALUES(3,'Outlet 3','Souvenir Store','Terminal 3',3);
INSERT INTO "RetailOutlets" VALUES(4,'Outlet 4','Souvenir Store','Concourse F',2);
INSERT INTO "RetailOutlets" VALUES(5,'Outlet 5','Bookshop','Terminal 2',2);
INSERT INTO "RetailOutlets" VALUES(6,'Outlet 6','Coffee Shop','Terminal 1',1);
INSERT INTO "RetailOutlets" VALUES(7,'Outlet 7','Restaurant','Terminal 3',3);
CREATE TABLE Terminals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airportId INTEGER,
    terminalName TEXT,
    FOREIGN KEY(airportId) REFERENCES Airports(id)
);
INSERT INTO "Terminals" VALUES(1,1,'Terminal 1');
INSERT INTO "Terminals" VALUES(2,1,'Terminal 2');
INSERT INTO "Terminals" VALUES(3,1,'Terminal 3');
INSERT INTO "Terminals" VALUES(4,2,'Terminal A');
INSERT INTO "Terminals" VALUES(5,2,'Terminal B');
INSERT INTO "Terminals" VALUES(6,3,'Terminal C');
INSERT INTO "Terminals" VALUES(7,3,'Terminal D');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Terminals',7);
INSERT INTO "sqlite_sequence" VALUES('Gates',9);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('RetailOutlets',7);
COMMIT;
