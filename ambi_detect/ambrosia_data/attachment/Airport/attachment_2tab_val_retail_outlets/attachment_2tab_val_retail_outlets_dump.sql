BEGIN TRANSACTION;
CREATE TABLE DutyFreeShops (
    ShopID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Size REAL,
    AnnualRevenue REAL
);
INSERT INTO "DutyFreeShops" VALUES(1,'Global Traveller','Terminal 3',12000.0,678900.0);
INSERT INTO "DutyFreeShops" VALUES(2,'Eastern Delicacies','Terminal 1',8000.0,545000.0);
CREATE TABLE Flights (
    FlightID INTEGER PRIMARY KEY AUTOINCREMENT,
    AirlineName TEXT,
    DepartureTime TIMESTAMP,
    ArrivalTime TIMESTAMP,
    OriginAirportCode CHAR(3),
    DestinationAirportCode CHAR(3)
);
INSERT INTO "Flights" VALUES(1,'United Airlines','2023-02-12 10:00:00','2023-02-12 13:30:00','JFK','LAX');
INSERT INTO "Flights" VALUES(2,'Delta Airlines','2023-02-12 14:00:00','2023-02-12 17:30:00','ATL','SEA');
CREATE TABLE PassengerCheckIns (
    CheckInID INTEGER PRIMARY KEY AUTOINCREMENT,
    FlightID INTEGER,
    PassengerName TEXT,
    BoardingPassNumber INTEGER,
    CheckInTime TIMESTAMP,
    FOREIGN KEY (FlightID) REFERENCES Flights(FlightID)
);
INSERT INTO "PassengerCheckIns" VALUES(1,1,'John Doe',1234,'2023-02-12 09:30:00');
INSERT INTO "PassengerCheckIns" VALUES(2,2,'Emma Smith',5678,'2023-02-12 10:30:00');
CREATE TABLE SouvenirStores (
    StoreID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    InventoryCount INTEGER,
    AveragePurchaseAmount REAL
);
INSERT INTO "SouvenirStores" VALUES(1,'Gift World','Terminal 3',500,25.0);
INSERT INTO "SouvenirStores" VALUES(2,'Memories of Journey','Terminal 2',700,20.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('DutyFreeShops',2);
INSERT INTO "sqlite_sequence" VALUES('SouvenirStores',2);
INSERT INTO "sqlite_sequence" VALUES('Flights',2);
INSERT INTO "sqlite_sequence" VALUES('PassengerCheckIns',2);
COMMIT;
