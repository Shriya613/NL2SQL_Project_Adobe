BEGIN TRANSACTION;
CREATE TABLE Airlines(
    id INTEGER PRIMARY KEY,
    airlineName TEXT);
INSERT INTO "Airlines" VALUES(1,'United Airlines');
INSERT INTO "Airlines" VALUES(2,'Delta Air Lines');
INSERT INTO "Airlines" VALUES(3,'Southwest Airlines');
INSERT INTO "Airlines" VALUES(4,'American Airlines');
INSERT INTO "Airlines" VALUES(5,'JetBlue Airways');
CREATE TABLE Available_locations(
    id INTEGER PRIMARY KEY,
    location TEXT);
INSERT INTO "Available_locations" VALUES(1,'Entrance A');
INSERT INTO "Available_locations" VALUES(2,'Departure Hall');
INSERT INTO "Available_locations" VALUES(3,'Gate C');
INSERT INTO "Available_locations" VALUES(4,'Baggage Claim');
INSERT INTO "Available_locations" VALUES(5,'Exit D');
CREATE TABLE Bookings(
    bookingID INTEGER PRIMARY KEY,
    customerName TEXT,
    flightId INTEGER REFERENCES Flights(flightID),
    luggageTrolleyId INTEGER,
    porterId INTEGER,
    FOREIGN KEY(luggageTrolleyId) REFERENCES Luggage_trolleys(trolleyID),
    FOREIGN KEY(porterId) REFERENCES Porters(porterID));
INSERT INTO "Bookings" VALUES(1,'John Doe',1,1,1);
INSERT INTO "Bookings" VALUES(2,'Jane Smith',2,NULL,NULL);
INSERT INTO "Bookings" VALUES(3,'Michael Johnson',3,2,2);
INSERT INTO "Bookings" VALUES(4,'Emily Davis',4,NULL,NULL);
INSERT INTO "Bookings" VALUES(5,'David Brown',5,1,1);
CREATE TABLE Flights(
    flightID INTEGER PRIMARY KEY,
    originLocation INTEGER REFERENCES Available_locations(id),
    destinationLocation INTEGER REFERENCES Available_locations(id),
    airlineId INTEGER REFERENCES Airlines(id));
INSERT INTO "Flights" VALUES(1,1,2,1);
INSERT INTO "Flights" VALUES(2,3,4,2);
INSERT INTO "Flights" VALUES(3,1,5,3);
INSERT INTO "Flights" VALUES(4,2,3,4);
INSERT INTO "Flights" VALUES(5,4,1,5);
CREATE TABLE "Luggage_trolleys"(
    trolleyID INTEGER PRIMARY KEY, "TrolleySerialNumber" INTEGER,
    availableLocation INTEGER REFERENCES Available_locations(id));
INSERT INTO "Luggage_trolleys" VALUES(1,123,1);
INSERT INTO "Luggage_trolleys" VALUES(2,234,2);
CREATE TABLE Porters(
    porterID INTEGER PRIMARY KEY,
    serviceType TEXT ,
    availableLocation INTEGER REFERENCES Available_locations(id));
INSERT INTO "Porters" VALUES(1,'Assistance',1);
INSERT INTO "Porters" VALUES(2,'Golf Cart Service',2);
COMMIT;
