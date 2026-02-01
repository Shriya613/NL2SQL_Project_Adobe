BEGIN TRANSACTION;
CREATE TABLE Airlines (
    airlineID INTEGER PRIMARY KEY,
    name TEXT,
    headquarters TEXT
);
INSERT INTO "Airlines" VALUES(1,'United Airlines','Chicago');
INSERT INTO "Airlines" VALUES(2,'British Airways','London');
INSERT INTO "Airlines" VALUES(3,'Air France','Paris');
CREATE TABLE Airports (
    airportID INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT,
    country TEXT
);
INSERT INTO "Airports" VALUES(1,'JFK International','New York','USA');
INSERT INTO "Airports" VALUES(2,'Heathrow','London','UK');
INSERT INTO "Airports" VALUES(3,'Charles de Gaulle','Paris','France');
CREATE TABLE Amenities (
    amenityID INTEGER PRIMARY KEY,
    description TEXT,
    availableInFrequentFlyerLounge BOOLEAN DEFAULT 0,
    availableInVIPLounge BOOLEAN DEFAULT 0
);
INSERT INTO "Amenities" VALUES(1,'Complimentary Wi-Fi',1,1);
INSERT INTO "Amenities" VALUES(2,'Power outlets at every seat',1,1);
CREATE TABLE Flights (
    flightID INTEGER PRIMARY KEY,
    originAirport INTEGER,
    destinationAirport INTEGER,
    departureTime DATETIME,
    arrivalTime DATETIME,
    airlineID INTEGER,
    FOREIGN KEY(originAirport) REFERENCES Airports(airportID),
    FOREIGN KEY(destinationAirport) REFERENCES Airports(airportID),
    FOREIGN KEY(airlineID) REFERENCES Airlines(airlineID)
);
INSERT INTO "Flights" VALUES(1,1,2,'2023-01-01 08:00:00','2023-01-01 10:00:00',1);
INSERT INTO "Flights" VALUES(2,1,3,'2023-01-01 12:00:00','2023-01-01 14:00:00',2);
INSERT INTO "Flights" VALUES(3,2,1,'2023-01-01 16:00:00','2023-01-01 18:00:00',1);
INSERT INTO "Flights" VALUES(4,2,3,'2023-01-01 20:00:00','2023-01-01 22:00:00',2);
INSERT INTO "Flights" VALUES(5,3,1,'2023-01-02 07:00:00','2023-01-02 09:00:00',1);
INSERT INTO "Flights" VALUES(6,3,2,'2023-01-02 11:00:00','2023-01-02 13:00:00',2);
CREATE TABLE "FrequentFlyerLounges" (
    loungeID INTEGER PRIMARY KEY,
    airportID INTEGER,
    amenityID INTEGER, capacity INTEGER,
    FOREIGN KEY(airportID) REFERENCES Airports(airportID),
    FOREIGN KEY(amenityID) REFERENCES Amenities(amenityID)
);
INSERT INTO "FrequentFlyerLounges" VALUES(1,1,1,250);
INSERT INTO "FrequentFlyerLounges" VALUES(2,2,2,150);
CREATE TABLE Passengers (
    passengerID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    frequentFlyerNumber INTEGER,
    vipStatus BOOLEAN
);
INSERT INTO "Passengers" VALUES(1,'John','Doe',12345,1);
INSERT INTO "Passengers" VALUES(2,'Jane','Smith',67890,0);
INSERT INTO "Passengers" VALUES(3,'Alice','Johnson',34567,1);
INSERT INTO "Passengers" VALUES(4,'Bob','Williams',23456,0);
INSERT INTO "Passengers" VALUES(5,'Eve','Martin',87654,1);
CREATE TABLE "VIPLounges" (
    loungeID INTEGER PRIMARY KEY,
    airportID INTEGER,
    amenityID INTEGER, capacity INTEGER,
    FOREIGN KEY(airportID) REFERENCES Airports(airportID),
    FOREIGN KEY(amenityID) REFERENCES Amenities(amenityID)
);
INSERT INTO "VIPLounges" VALUES(1,1,1,40);
INSERT INTO "VIPLounges" VALUES(2,2,2,50);
COMMIT;
