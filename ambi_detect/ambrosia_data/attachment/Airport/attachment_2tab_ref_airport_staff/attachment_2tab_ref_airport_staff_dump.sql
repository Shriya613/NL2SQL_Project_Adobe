BEGIN TRANSACTION;
CREATE TABLE Flights(
    flightNumber INT PRIMARY KEY,
    departureTime DATETIME,
    arrivalTime DATETIME,
    pilotId INTEGER REFERENCES Pilots(id),
    gateAgentId INTEGER REFERENCES GateAgents(id));
INSERT INTO "Flights" VALUES(101,'2022-01-01 08:00','2022-01-01 10:00',1,1);
INSERT INTO "Flights" VALUES(102,'2022-01-01 09:00','2022-01-01 11:00',2,2);
INSERT INTO "Flights" VALUES(103,'2022-01-01 10:00','2022-01-01 12:00',3,3);
INSERT INTO "Flights" VALUES(104,'2022-01-01 11:00','2022-01-01 13:00',4,4);
INSERT INTO "Flights" VALUES(105,'2022-01-01 12:00','2022-01-01 14:00',5,5);
CREATE TABLE GateAgents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    gateNumber INT,
    speaksId INTEGER REFERENCES Languages(id),
    FOREIGN KEY (speaksId) REFERENCES Languages(id));
INSERT INTO "GateAgents" VALUES(1,'Paul White','A1',1);
INSERT INTO "GateAgents" VALUES(2,'Tom Black','C5',2);
INSERT INTO "GateAgents" VALUES(3,'Lucy Blue','D12',3);
INSERT INTO "GateAgents" VALUES(4,'Kelly Red','E14',4);
INSERT INTO "GateAgents" VALUES(5,'Hannah Pink','F20',5);
CREATE TABLE Languages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language TEXT UNIQUE);
INSERT INTO "Languages" VALUES(1,'English');
INSERT INTO "Languages" VALUES(2,'French');
INSERT INTO "Languages" VALUES(3,'German');
INSERT INTO "Languages" VALUES(4,'Japanese');
INSERT INTO "Languages" VALUES(5,'Russian');
CREATE TABLE Passengers(
    passengerId INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    age INT );
INSERT INTO "Passengers" VALUES(1,'Alexander','Smith',25);
INSERT INTO "Passengers" VALUES(2,'Emma','Williams',30);
INSERT INTO "Passengers" VALUES(3,'David','Jones',40);
INSERT INTO "Passengers" VALUES(4,'Daniel','Black',50);
INSERT INTO "Passengers" VALUES(5,'Laura','White',60);
CREATE TABLE Pilots(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    licenseNumber INT,
    speaksId INTEGER REFERENCES Languages(id),
    FOREIGN KEY (speaksId) REFERENCES Languages(id));
INSERT INTO "Pilots" VALUES(1,'John Doe',1234567890,1);
INSERT INTO "Pilots" VALUES(2,'Mary Smith',987654321,2);
INSERT INTO "Pilots" VALUES(3,'Alice Johnson',5555555555,3);
INSERT INTO "Pilots" VALUES(4,'Bob Brown',4444444444,4);
INSERT INTO "Pilots" VALUES(5,'Charlie Green',3333333333,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Languages',5);
INSERT INTO "sqlite_sequence" VALUES('Pilots',5);
INSERT INTO "sqlite_sequence" VALUES('GateAgents',5);
INSERT INTO "sqlite_sequence" VALUES('Passengers',5);
COMMIT;
