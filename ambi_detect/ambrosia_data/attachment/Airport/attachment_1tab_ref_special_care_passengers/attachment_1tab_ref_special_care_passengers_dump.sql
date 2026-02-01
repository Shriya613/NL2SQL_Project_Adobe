BEGIN TRANSACTION;
CREATE TABLE Airlines (
    AirlineID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name VARCHAR(255),
    HeadquartersCity VARCHAR(255),
    HomeAirportId INTEGER,
    FOREIGN KEY (HomeAirportId) REFERENCES Airports(AirportID)
);
INSERT INTO "Airlines" VALUES(1,'American Airlines','Fort Worth',1);
INSERT INTO "Airlines" VALUES(2,'United Airlines','Chicago',2);
INSERT INTO "Airlines" VALUES(3,'Air Canada','Montreal',NULL);
INSERT INTO "Airlines" VALUES(4,'Air France','Tremblay-en-France',4);
INSERT INTO "Airlines" VALUES(5,'Cathay Pacific','Hong Kong',5);
CREATE TABLE Airports (
    AirportID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name VARCHAR(255),
    City VARCHAR(255),
    Country VARCHAR(255)
);
INSERT INTO "Airports" VALUES(1,'John F. Kennedy International Airport','New York','USA');
INSERT INTO "Airports" VALUES(2,'Chicago O''Hare International Airport','Chicago','USA');
INSERT INTO "Airports" VALUES(3,'Toronto Pearson International Airport','Toronto','Canada');
INSERT INTO "Airports" VALUES(4,'Charles de Gaulle Airport','Paris','France');
INSERT INTO "Airports" VALUES(5,'Hong Kong International Airport','Hong Kong','Hong Kong');
CREATE TABLE Assistants (
    AssistantID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name VARCHAR(255)
);
INSERT INTO "Assistants" VALUES(1,'Claire Allen');
INSERT INTO "Assistants" VALUES(2,'Emma Thompson');
INSERT INTO "Assistants" VALUES(3,'Jack Williams');
INSERT INTO "Assistants" VALUES(4,'Lucy Jones');
INSERT INTO "Assistants" VALUES(5,'Michael Johnson');
CREATE TABLE Flights (
    FlightID INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureTime TIME,
    ArrivalTime TIME,
    Origin VARCHAR(255),
    Destination VARCHAR(255)
);
INSERT INTO "Flights" VALUES(1,'10:00:00','13:00:00','JFK','LAX');
INSERT INTO "Flights" VALUES(2,'14:00:00','17:00:00','ORD','ATL');
INSERT INTO "Flights" VALUES(3,'06:00:00','09:00:00','YYZ','YUL');
INSERT INTO "Flights" VALUES(4,'18:00:00','21:00:00','CDG','DUB');
INSERT INTO "Flights" VALUES(5,'09:00:00','12:00:00','PEK','HKG');
CREATE TABLE PassengerAssistanceRequests (
    RequestID INTEGER PRIMARY KEY AUTOINCREMENT,
    PassengerID INTEGER,
    AssistantID INTEGER,
    Type VARCHAR(100) ,
    FOREIGN KEY (PassengerID) REFERENCES Passengers(PassengerID),
    FOREIGN KEY (AssistantID) REFERENCES Assistants(AssistantID)
);
INSERT INTO "PassengerAssistanceRequests" VALUES(1,1,1,'Unaccompanied Minor');
INSERT INTO "PassengerAssistanceRequests" VALUES(2,2,2,'Unaccompanied Minor');
INSERT INTO "PassengerAssistanceRequests" VALUES(3,3,1,'Tailored Assistance');
INSERT INTO "PassengerAssistanceRequests" VALUES(4,4,3,'Tailored Assistance');
INSERT INTO "PassengerAssistanceRequests" VALUES(5,5,4,'General Assistance');
CREATE TABLE Passengers (
    PassengerID INTEGER PRIMARY KEY AUTOINCREMENT,
    DateOfBirth DATE, FullName TEXT);
INSERT INTO "Passengers" VALUES(1,'1990-01-15','Alice Smith');
INSERT INTO "Passengers" VALUES(2,'1985-02-20','Bob Johnson');
INSERT INTO "Passengers" VALUES(3,'1992-03-25','Carol Williams');
INSERT INTO "Passengers" VALUES(4,'1988-04-05','David Brown');
INSERT INTO "Passengers" VALUES(5,'1994-05-10','Eva Miller');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Passengers',5);
INSERT INTO "sqlite_sequence" VALUES('Assistants',5);
INSERT INTO "sqlite_sequence" VALUES('PassengerAssistanceRequests',5);
COMMIT;
