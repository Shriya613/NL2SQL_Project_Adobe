BEGIN TRANSACTION;
CREATE TABLE Airlines (
    id INTEGER PRIMARY KEY,
    name TEXT,
    headquarterCity TEXT,
    website TEXT
);
INSERT INTO "Airlines" VALUES(1,'British Airways','London','www.ba.com');
INSERT INTO "Airlines" VALUES(2,'American Airlines','Dallas','www.aa.com');
INSERT INTO "Airlines" VALUES(3,'Delta Airlines','Atlanta','www.delta.com');
INSERT INTO "Airlines" VALUES(4,'United Airlines','Chicago','www.united.com');
CREATE TABLE Airports (
    id INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT,
    country TEXT
);
INSERT INTO "Airports" VALUES(1,'JFK International','New York','USA');
INSERT INTO "Airports" VALUES(2,'Heathrow','London','UK');
CREATE TABLE Flight (
    flightNumber INTEGER PRIMARY KEY,
    departureTime DATETIME,
    arrivalTime DATETIME,
    destinationId INTEGER,
    airlineId INTEGER,
    FOREIGN KEY(destinationId) REFERENCES Airports(id),
    FOREIGN KEY(airlineId) REFERENCES Airlines(id)
);
INSERT INTO "Flight" VALUES(101,'2022-10-29 08:00:00','2022-10-29 10:00:00',1,1);
INSERT INTO "Flight" VALUES(102,'2022-10-29 16:00:00','2022-10-29 17:00:00',2,2);
CREATE TABLE GateAgents (
    agentID INTEGER PRIMARY KEY,
    staffID INTEGER,
    FOREIGN KEY(staffID) REFERENCES Staff(staffID)
);
INSERT INTO "GateAgents" VALUES(1,1);
INSERT INTO "GateAgents" VALUES(2,2);
CREATE TABLE Languages (
    id INTEGER PRIMARY KEY,
    language TEXT
);
INSERT INTO "Languages" VALUES(1,'Spanish');
INSERT INTO "Languages" VALUES(2,'English');
CREATE TABLE Pilots (
    pilotID INTEGER PRIMARY KEY,
    staffID INTEGER,
    FOREIGN KEY(staffID) REFERENCES Staff(staffID)
);
INSERT INTO "Pilots" VALUES(1,3);
INSERT INTO "Pilots" VALUES(2,4);
CREATE TABLE Staff (
    staffID INTEGER PRIMARY KEY,
    role TEXT,
    airportID INTEGER,
    languageID INTEGER, FullName TEXT,
    FOREIGN KEY(airportID) REFERENCES Airports(id),
    FOREIGN KEY(languageID) REFERENCES Languages(id)
);
INSERT INTO "Staff" VALUES(1,'Gate Agent',1,1,'John Smith');
INSERT INTO "Staff" VALUES(2,'Gate Agent',1,2,'Emily Doe');
INSERT INTO "Staff" VALUES(3,'Pilot',1,1,'Michael Jackson');
INSERT INTO "Staff" VALUES(4,'Pilot',1,2,'Paula Abdul');
COMMIT;
