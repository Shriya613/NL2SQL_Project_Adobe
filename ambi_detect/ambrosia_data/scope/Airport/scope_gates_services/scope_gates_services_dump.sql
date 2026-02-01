BEGIN TRANSACTION;
CREATE TABLE Airlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    country TEXT
);
INSERT INTO "Airlines" VALUES(1,'United Airlines','USA');
INSERT INTO "Airlines" VALUES(2,'Delta Airlines','USA');
INSERT INTO "Airlines" VALUES(3,'American Airlines','USA');
INSERT INTO "Airlines" VALUES(4,'Southwest Airlines','USA');
INSERT INTO "Airlines" VALUES(5,'Air Canada','Canada');
CREATE TABLE Airports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airportName TEXT,
    city TEXT,
    country TEXT
);
INSERT INTO "Airports" VALUES(1,'John F. Kennedy International Airport','New York','USA');
INSERT INTO "Airports" VALUES(2,'Hartsfield-Jackson Atlanta International Airport','Atlanta','USA');
INSERT INTO "Airports" VALUES(3,'Los Angeles International Airport','Los Angeles','USA');
INSERT INTO "Airports" VALUES(4,'Toronto Pearson International Airport','Toronto','Canada');
INSERT INTO "Airports" VALUES(5,'Vancouver International Airport','Vancouver','Canada');
CREATE TABLE Flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flightNumber TEXT,
    departureTime TIMESTAMP,
    arrivalTime TIMESTAMP,
    airlineId INTEGER,
    originAirportId INTEGER,
    destinationAirportId INTEGER,
    FOREIGN KEY(airlineId) REFERENCES Airlines(id),
    FOREIGN KEY(originAirportId) REFERENCES Airports(id),
    FOREIGN KEY(destinationAirportId) REFERENCES Airports(id)
);
INSERT INTO "Flights" VALUES(1,'UA123','2022-07-10 08:00:00','2022-07-10 10:00:00',1,1,2);
INSERT INTO "Flights" VALUES(2,'DL456','2022-07-10 10:00:00','2022-07-10 12:00:00',2,1,3);
INSERT INTO "Flights" VALUES(3,'AA789','2022-07-10 14:00:00','2022-07-10 16:00:00',3,1,4);
INSERT INTO "Flights" VALUES(4,'SW101','2022-07-10 16:00:00','2022-07-10 18:00:00',4,1,5);
INSERT INTO "Flights" VALUES(5,'AC246','2022-07-10 18:00:00','2022-07-10 20:00:00',5,1,6);
CREATE TABLE Gates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gateNumber INTEGER,
    terminal INTEGER,
    airportId INTEGER,
    FOREIGN KEY(airportId) REFERENCES Airports(id)
);
INSERT INTO "Gates" VALUES(1,1,1,1);
INSERT INTO "Gates" VALUES(2,2,1,1);
INSERT INTO "Gates" VALUES(3,3,1,1);
INSERT INTO "Gates" VALUES(4,4,1,1);
INSERT INTO "Gates" VALUES(5,5,1,1);
CREATE TABLE Gates_Services (
    gateId INTEGER,
    serviceId INTEGER,
    PRIMARY KEY(gateId, serviceId),
    FOREIGN KEY(gateId) REFERENCES Gates(id),
    FOREIGN KEY(serviceId) REFERENCES Services(id)
);
INSERT INTO "Gates_Services" VALUES(1,1);
INSERT INTO "Gates_Services" VALUES(2,1);
INSERT INTO "Gates_Services" VALUES(3,1);
INSERT INTO "Gates_Services" VALUES(4,1);
INSERT INTO "Gates_Services" VALUES(5,1);
INSERT INTO "Gates_Services" VALUES(5,3);
CREATE TABLE Services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    serviceName TEXT
);
INSERT INTO "Services" VALUES(1,'Baggage_Claim');
INSERT INTO "Services" VALUES(2,'Food_Court');
INSERT INTO "Services" VALUES(3,'Duty_Free');
INSERT INTO "Services" VALUES(4,'Car_Rental');
INSERT INTO "Services" VALUES(5,'Lost_and_Found');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Gates',5);
INSERT INTO "sqlite_sequence" VALUES('Services',5);
COMMIT;
