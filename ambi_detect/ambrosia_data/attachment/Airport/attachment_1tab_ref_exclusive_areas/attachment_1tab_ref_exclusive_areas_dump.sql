BEGIN TRANSACTION;
CREATE TABLE AircraftType(
    id INTEGER PRIMARY KEY,
    typeName TEXT);
INSERT INTO "AircraftType" VALUES(1,'Boeing 737');
INSERT INTO "AircraftType" VALUES(2,'Airbus A320');
INSERT INTO "AircraftType" VALUES(3,'Embraer E190');
INSERT INTO "AircraftType" VALUES(4,'Bombardier Q400');
INSERT INTO "AircraftType" VALUES(5,'De Havilland Canada Dash 8');
CREATE TABLE Amenity(
    amenityId INTEGER PRIMARY KEY AUTOINCREMENT,
    amenityValue TEXT);
INSERT INTO "Amenity" VALUES(1,'Complimentary Wi-Fi');
INSERT INTO "Amenity" VALUES(2,'Power Outlets');
INSERT INTO "Amenity" VALUES(3,'Charging Stations');
INSERT INTO "Amenity" VALUES(4,'Restrooms');
INSERT INTO "Amenity" VALUES(5,'Luggage Storage');
CREATE TABLE AreaAmenityMap(
    areaId INTEGER,
    amenityId INTEGER,
    FOREIGN KEY (areaId) REFERENCES ExclusiveArea(areaId),
    FOREIGN KEY (amenityId) REFERENCES Amenity(amenityId));
INSERT INTO "AreaAmenityMap" VALUES(1,1);
INSERT INTO "AreaAmenityMap" VALUES(1,2);
INSERT INTO "AreaAmenityMap" VALUES(2,3);
INSERT INTO "AreaAmenityMap" VALUES(3,4);
INSERT INTO "AreaAmenityMap" VALUES(4,5);
INSERT INTO "AreaAmenityMap" VALUES(5,1);
CREATE TABLE ExclusiveArea(
    areaId INTEGER PRIMARY KEY AUTOINCREMENT,
    terminalId INTEGER,
    exclusiveAreaValue TEXT,
    FOREIGN KEY (terminalId) REFERENCES Terminal(id));
INSERT INTO "ExclusiveArea" VALUES(1,1,'Frequent Flyer Lounge');
INSERT INTO "ExclusiveArea" VALUES(2,2,'Business Class Lounge');
INSERT INTO "ExclusiveArea" VALUES(3,3,'First Class Lounge');
INSERT INTO "ExclusiveArea" VALUES(4,4,'Frequent Flyer Lounge');
INSERT INTO "ExclusiveArea" VALUES(5,5,'VIP Lounge');
CREATE TABLE Flight(
    flightNumber INTEGER PRIMARY KEY,
    aircraftTypeId INTEGER,
    departureTime TIMESTAMP,
    arrivalTime TIMESTAMP,
    originTerminalId INTEGER,
    destinationTerminalId INTEGER,
    FOREIGN KEY (aircraftTypeId) REFERENCES AircraftType(id),
    FOREIGN KEY (originTerminalId) REFERENCES Terminal(id),
    FOREIGN KEY (destinationTerminalId) REFERENCES Terminal(id));
INSERT INTO "Flight" VALUES(1,1,'2022-01-01 06:00:00','2022-01-01 08:00:00',1,4);
INSERT INTO "Flight" VALUES(2,2,'2022-01-01 08:30:00','2022-01-01 10:30:00',2,3);
INSERT INTO "Flight" VALUES(3,3,'2022-01-01 11:00:00','2022-01-01 13:00:00',3,5);
INSERT INTO "Flight" VALUES(4,4,'2022-01-01 13:30:00','2022-01-01 15:30:00',4,2);
INSERT INTO "Flight" VALUES(5,5,'2022-01-01 16:00:00','2022-01-01 18:00:00',5,1);
CREATE TABLE Gate(
    id INTEGER PRIMARY KEY,
    terminal_id INTEGER,
    gate_number INTEGER,
    FOREIGN KEY (terminal_id) REFERENCES Terminal(id));
INSERT INTO "Gate" VALUES(1,1,7);
INSERT INTO "Gate" VALUES(2,1,14);
INSERT INTO "Gate" VALUES(3,2,25);
INSERT INTO "Gate" VALUES(4,2,38);
INSERT INTO "Gate" VALUES(5,3,50);
CREATE TABLE Terminal(
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT);
INSERT INTO "Terminal" VALUES(1,'Terminal A','Gate 1-20');
INSERT INTO "Terminal" VALUES(2,'Terminal B','Gate 21-39');
INSERT INTO "Terminal" VALUES(3,'Terminal C','Gate 40-60');
INSERT INTO "Terminal" VALUES(4,'Terminal D','Gate 61-80');
INSERT INTO "Terminal" VALUES(5,'Terminal E','Gate 81-100');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('ExclusiveArea',5);
INSERT INTO "sqlite_sequence" VALUES('Amenity',5);
COMMIT;
