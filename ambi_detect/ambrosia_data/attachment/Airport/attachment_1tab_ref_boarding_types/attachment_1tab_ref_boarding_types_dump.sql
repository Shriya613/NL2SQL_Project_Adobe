BEGIN TRANSACTION;
CREATE TABLE Airlines (
    AirlineID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Airlines" VALUES(1,'Alaskan Airlines');
INSERT INTO "Airlines" VALUES(2,'United Airlines');
INSERT INTO "Airlines" VALUES(3,'Delta Airlines');
INSERT INTO "Airlines" VALUES(4,'Southwest Airlines');
INSERT INTO "Airlines" VALUES(5,'American Airlines');
CREATE TABLE Flights (
    FlightID INTEGER PRIMARY KEY AUTOINCREMENT,
    Origin TEXT,
    Destination TEXT,
    DepartureTime DATETIME,
    ArrivalTime DATETIME,
    AirlineID INTEGER,
    PlaneID INTEGER,
    FOREIGN KEY(AirlineID) REFERENCES Airlines(AirlineID),
    FOREIGN KEY(PlaneID) REFERENCES Planes(PlaneID)
);
INSERT INTO "Flights" VALUES(1,'Seattle','New York','2023-04-01 10:00:00','2023-04-01 14:00:00',1,1);
INSERT INTO "Flights" VALUES(2,'Los Angeles','Chicago','2023-04-01 12:00:00','2023-04-01 16:00:00',2,2);
INSERT INTO "Flights" VALUES(3,'San Francisco','Houston','2023-04-01 14:00:00','2023-04-01 18:00:00',3,3);
INSERT INTO "Flights" VALUES(4,'Dallas','Atlanta','2023-04-01 16:00:00','2023-04-01 20:00:00',4,4);
INSERT INTO "Flights" VALUES(5,'Miami','Philadelphia','2023-04-01 18:00:00','2023-04-01 22:00:00',5,5);
CREATE TABLE Passengers (
    PassengerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT,
    BoardingType TEXT,
    FlightID INTEGER,
    FOREIGN KEY(FlightID) REFERENCES Flights(FlightID)
);
INSERT INTO "Passengers" VALUES(1,'John Doe',35,'Priority Boarding',1);
INSERT INTO "Passengers" VALUES(2,'Jane Smith',28,'Family Boarding',1);
INSERT INTO "Passengers" VALUES(3,'Emily Johnson',32,'Priority Boarding',2);
INSERT INTO "Passengers" VALUES(4,'Robert Brown',40,'Family Boarding',2);
INSERT INTO "Passengers" VALUES(5,'Tom Wilson',25,'Priority Boarding',3);
INSERT INTO "Passengers" VALUES(6,'Linda Davis',30,'Family Boarding',3);
INSERT INTO "Passengers" VALUES(7,'David Miller',22,'Priority Boarding',4);
INSERT INTO "Passengers" VALUES(8,'Sarah Taylor',28,'Family Boarding',4);
INSERT INTO "Passengers" VALUES(9,'Mark Lee',35,'Priority Boarding',5);
INSERT INTO "Passengers" VALUES(10,'Laura Kim',30,'Family Boarding',5);
CREATE TABLE Planes (
    PlaneID INTEGER PRIMARY KEY AUTOINCREMENT,
    Model TEXT,
    Capacity INT,
    FOREIGN KEY(PlaneID) REFERENCES Flights(PlaneID)
);
INSERT INTO "Planes" VALUES(1,'Boeing 737',200);
INSERT INTO "Planes" VALUES(2,'Airbus A380',519);
INSERT INTO "Planes" VALUES(3,'Embraer E190',100);
INSERT INTO "Planes" VALUES(4,'Bombardier CRJ900',100);
INSERT INTO "Planes" VALUES(5,'Boeing 747',600);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Planes',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Passengers',10);
COMMIT;
