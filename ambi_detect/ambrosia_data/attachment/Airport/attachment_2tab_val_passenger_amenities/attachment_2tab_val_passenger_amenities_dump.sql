BEGIN TRANSACTION;
CREATE TABLE Flight(
    id INTEGER PRIMARY KEY,
    FlightNumber TEXT,
    DepartureTime TIMESTAMP,
    Destination TEXT,
    AircraftModel TEXT
);
INSERT INTO "Flight" VALUES(1,'AA123','2023-04-01 07:00:00','New York JFK','Boeing 787 Dreamliner');
INSERT INTO "Flight" VALUES(2,'UA999','2023-04-01 10:00:00','Los Angeles LAX','Airbus A320');
INSERT INTO "Flight" VALUES(3,'DL456','2023-04-01 13:00:00','Chicago ORD','Embraer E190');
INSERT INTO "Flight" VALUES(4,'WN888','2023-04-01 16:00:00','Miami MIA','Bombardier CRJ-700');
INSERT INTO "Flight" VALUES(5,'BA100','2023-04-01 19:00:00','London Heathrow LHR','Boing 747-400');
CREATE TABLE Gate(
    id INTEGER PRIMARY KEY,
    GateNumber INTEGER,
    TerminalName TEXT,
    AirlinesCode TEXT
);
INSERT INTO "Gate" VALUES(1,10,'Terminal A','AA');
INSERT INTO "Gate" VALUES(2,20,'Terminal B','UA');
INSERT INTO "Gate" VALUES(3,30,'Terminal C','DL');
INSERT INTO "Gate" VALUES(4,40,'Terminal D','WN');
INSERT INTO "Gate" VALUES(5,50,'Terminal E','BA');
CREATE TABLE Lounge(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Capacity INTEGER,
    Accessibility TEXT
);
INSERT INTO "Lounge" VALUES(1,'Premium Lounge','Terminal A',200,'All');
INSERT INTO "Lounge" VALUES(2,'Business Class Lounge','Terminal B',300,'Wheelchair Users');
INSERT INTO "Lounge" VALUES(3,'Family Lounge','Terminal C',400,'All');
INSERT INTO "Lounge" VALUES(4,'VIP Lounge','Terminal D',500,'Wheelchair Users');
INSERT INTO "Lounge" VALUES(5,'General Lounge','Terminal E',600,'All');
CREATE TABLE ObservationDeck(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    ViewType TEXT,
    Accessibility TEXT
);
INSERT INTO "ObservationDeck" VALUES(1,'Rooftop Deck','Near Terminal A','Panoramic view','All');
INSERT INTO "ObservationDeck" VALUES(2,'Windowside Deck','By Gate 10 at Terminal B','Tarmac views','Wheelchair Users');
INSERT INTO "ObservationDeck" VALUES(3,'Elevated Deck','Between Terminals C and D','Full airport view','All');
INSERT INTO "ObservationDeck" VALUES(4,'Landing Deck','At end of Pier at Terminal E','Approaching aircrafts','Wheelchair Users');
INSERT INTO "ObservationDeck" VALUES(5,'Waterfront Deck','Close to parking lot','City skyline and water views','All');
CREATE TABLE SecurityCheckpoint(
    id INTEGER PRIMARY KEY,
    CheckpointLocation TEXT,
    Accessibility TEXT,
    OperatingHours TEXT
);
INSERT INTO "SecurityCheckpoint" VALUES(1,'Main checkpoint near entrance','All','05:00 - 22:00');
INSERT INTO "SecurityCheckpoint" VALUES(2,'International Arrivals checkpoint','Wheelchair Users','06:00 - 21:00');
INSERT INTO "SecurityCheckpoint" VALUES(3,'Domestic Departures checkpoint','All','04:30 - 23:00');
INSERT INTO "SecurityCheckpoint" VALUES(4,'Express checkpoint for elite members','All','05:00 - 21:00');
INSERT INTO "SecurityCheckpoint" VALUES(5,'Private checkpoint for VIP passengers','All','06:00 - 22:00');
COMMIT;
