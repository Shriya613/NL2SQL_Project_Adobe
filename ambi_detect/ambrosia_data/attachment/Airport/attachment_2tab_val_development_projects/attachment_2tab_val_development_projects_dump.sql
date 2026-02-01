BEGIN TRANSACTION;
CREATE TABLE Airlines(
    id INTEGER PRIMARY KEY,
    AirlineName TEXT,
    HeadquartersLocation TEXT,
    FleetSize INTEGER,
    AirportID INTEGER,
    FOREIGN KEY(AirportID) REFERENCES Airports(id));
INSERT INTO "Airlines" VALUES(1,'American Airlines','Fort Worth, Texas',950,1);
INSERT INTO "Airlines" VALUES(2,'British Airways','Watford, Hertfordshire',280,2);
INSERT INTO "Airlines" VALUES(3,'Qantas Airways','Mascot, New South Wales',180,3);
INSERT INTO "Airlines" VALUES(4,'Cathay Pacific','Hong Kong',140,4);
INSERT INTO "Airlines" VALUES(5,'Lufthansa','Cologne',300,1);
INSERT INTO "Airlines" VALUES(6,'Emirates','Garhoud, Dubai',250,2);
INSERT INTO "Airlines" VALUES(7,'Aeromexico','Miguel Hidalgo',130,3);
INSERT INTO "Airlines" VALUES(8,'Ethiopian Airlines','Addis Ababa',110,4);
CREATE TABLE Airports(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    YearEstablished INTEGER);
INSERT INTO "Airports" VALUES(1,'JFK International','New York City',1948);
INSERT INTO "Airports" VALUES(2,'Dubai International','Dubai',1960);
INSERT INTO "Airports" VALUES(3,'Heathrow Airport','London',1946);
INSERT INTO "Airports" VALUES(4,'Charles de Gaulle Airport','Paris',1974);
INSERT INTO "Airports" VALUES(5,'Hong Kong International Airport','Hong Kong',1998);
CREATE TABLE Expansions(
    id INTEGER PRIMARY KEY,
    AirportID INTEGER,
    Airport_Expansion TEXT,
    Increased_Capacity TEXT,
    FOREIGN KEY(AirportID) REFERENCES Airports(id));
INSERT INTO "Expansions" VALUES(1,1,'Terminal 8 construction','2000 Passengers per day');
INSERT INTO "Expansions" VALUES(2,2,'Concourse D expansion','3000 Passengers per day');
INSERT INTO "Expansions" VALUES(3,3,'Northwest Terminal extension','1500 Passengers per day');
INSERT INTO "Expansions" VALUES(4,4,'Runway extension project','2500 Passengers per day');
CREATE TABLE Flights(
    id INTEGER PRIMARY KEY,
    FlightNumber TEXT,
    DepartureTime TIME,
    ArrivalTime TIME,
    AirportID INTEGER,
    FOREIGN KEY(AirportID) REFERENCES Airports(id));
INSERT INTO "Flights" VALUES(1,'AA123','06:00:00','08:00:00',1);
INSERT INTO "Flights" VALUES(2,'BA456','08:00:00','10:00:00',2);
INSERT INTO "Flights" VALUES(3,'QF789','10:00:00','12:00:00',3);
INSERT INTO "Flights" VALUES(4,'CX123','12:00:00','14:00:00',4);
INSERT INTO "Flights" VALUES(5,'LH456','14:00:00','16:00:00',1);
INSERT INTO "Flights" VALUES(6,'MU789','16:00:00','18:00:00',2);
INSERT INTO "Flights" VALUES(7,'AE998','18:00:00','20:00:00',3);
INSERT INTO "Flights" VALUES(8,'ET123','20:00:00','22:00:00',4);
CREATE TABLE Renovations(
    id INTEGER PRIMARY KEY,
    AirportID INTEGER,
    Renovation TEXT,
    Increased_Capacity TEXT,
    FOREIGN KEY(AirportID) REFERENCES Airports(id));
INSERT INTO "Renovations" VALUES(1,1,'Energy efficient lighting system','1000 Passengers per day');
INSERT INTO "Renovations" VALUES(2,2,'Baggage handling system upgrade','1000 Passengers per day');
INSERT INTO "Renovations" VALUES(3,3,'Security checkpoint modernization','500 Passengers per day');
INSERT INTO "Renovations" VALUES(4,4,'Passenger lounges renovation','1500 Passengers per day');
COMMIT;
