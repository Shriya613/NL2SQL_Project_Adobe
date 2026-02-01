BEGIN TRANSACTION;
CREATE TABLE Airlines (
    Airlines_Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name VARCHAR(100),
    Callsign VARCHAR(100),
    Headquarters VARCHAR(100));
INSERT INTO "Airlines" VALUES(1,'United Airlines','UAL','Chicago');
INSERT INTO "Airlines" VALUES(2,'Delta Air Lines','DL','Atlanta');
INSERT INTO "Airlines" VALUES(3,'American Airlines','AA','Fort Worth');
INSERT INTO "Airlines" VALUES(4,'Southwest Airlines','WN','Dallas');
INSERT INTO "Airlines" VALUES(5,'Alaska Airlines','AS','Seattle');
CREATE TABLE Airports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    City TEXT,
    Coordinates TEXT,
    IATA_Code CHAR(3),
    ICAO_Code CHAR(4),
    Altitude INT,
    FOREIGN KEY (IATA_Code) REFERENCES Flights(IATA_Departure),
    FOREIGN KEY (ICAO_Code) REFERENCES Flights(ICAO_Arrival));
INSERT INTO "Airports" VALUES(1,'New York','40.641798,-73.942208','JFK','KJFK',12);
INSERT INTO "Airports" VALUES(2,'Los Angeles','34.052235,-118.243683','LAX','KLAX',30);
INSERT INTO "Airports" VALUES(3,'Chicago','41.878113,-87.629799','ORD','KORD',601);
INSERT INTO "Airports" VALUES(4,'Dallas','32.782812,-96.802623','DFW','KDFW',465);
INSERT INTO "Airports" VALUES(5,'Miami','25.761703,-80.191797','MIA','KMIA',10);
CREATE TABLE Employees (
    Employee_Id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName VARCHAR(100),
    LastName VARCHAR(100),
    DateOfBirth DATE,
    Email VARCHAR(100),
    PhoneNumber VARCHAR(15),
    HireDate DATE);
INSERT INTO "Employees" VALUES(1,'John','Smith','1980-06-15','john.smith@example.com','555-1234','2005-05-15');
INSERT INTO "Employees" VALUES(2,'Jane','Doe','1990-08-20','jane.doe@example.com','555-5678','2006-06-15');
INSERT INTO "Employees" VALUES(3,'Michael','Jackson','1985-10-30','michael.jackson@example.com','555-9012','2007-07-15');
INSERT INTO "Employees" VALUES(4,'Emily','Williams','1995-12-05','emily.williams@example.com','555-1357','2008-08-15');
INSERT INTO "Employees" VALUES(5,'David','Martin','1988-02-12','david.martin@example.com','555-1415','2009-09-15');
CREATE TABLE Flights (
    Flight_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Departure_Time TIMESTAMP,
    Arrival_Time TIMESTAMP,
    Duration INT,
    Seats INT,
    IATA_Departure CHAR(3),
    ICAO_Arrival CHAR(4),
    Price DECIMAL(5,2),
    ForeignKey_Airline INTEGER,
    FOREIGN KEY (IATA_Departure) REFERENCES Airports(IATA_Code),
    FOREIGN KEY (ICAO_Arrival) REFERENCES Airports(ICAO_Code),
    FOREIGN KEY (ForeignKey_Airline) REFERENCES Airlines(Airlines_Id));
INSERT INTO "Flights" VALUES(1,'2023-01-01 08:00:00','2023-01-01 12:00:00',180,250,'JFK','KATL',250,1);
INSERT INTO "Flights" VALUES(2,'2023-01-02 09:00:00','2023-01-02 13:00:00',180,250,'LAX','KSFO',260,2);
INSERT INTO "Flights" VALUES(3,'2023-01-03 10:00:00','2023-01-03 14:00:00',180,250,'ORD','KPHX',240,3);
INSERT INTO "Flights" VALUES(4,'2023-01-04 11:00:00','2023-01-04 15:00:00',180,250,'DFW','KEWR',250,4);
INSERT INTO "Flights" VALUES(5,'2023-01-05 12:00:00','2023-01-05 16:00:00',180,250,'MIA','KSEA',260,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Airlines',5);
INSERT INTO "sqlite_sequence" VALUES('Employees',5);
COMMIT;
