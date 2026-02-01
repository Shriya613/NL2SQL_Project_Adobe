BEGIN TRANSACTION;
CREATE TABLE Airlines(
    id INTEGER PRIMARY KEY,
    airline_name TEXT,
    terminal_id INTEGER,
    FOREIGN KEY (terminal_id) REFERENCES Terminals(id));
INSERT INTO "Airlines" VALUES(1,'United Airlines',1);
INSERT INTO "Airlines" VALUES(2,'Delta Airlines',2);
CREATE TABLE Airports(
    code CHAR(2) PRIMARY KEY,
    name TEXT,
    city TEXT,
    country TEXT);
INSERT INTO "Airports" VALUES('LAX','Los Angeles International Airport','Los Angeles','USA');
INSERT INTO "Airports" VALUES('ORD','O''Hare International Airport','Chicago','USA');
CREATE TABLE Baggage_Claims(
    claim_number INT,
    carousel_location TEXT,
    airport_code CHAR(2),
    FOREIGN KEY (airport_code) REFERENCES Airports(code));
INSERT INTO "Baggage_Claims" VALUES(101,'Carousel 3','LAX');
INSERT INTO "Baggage_Claims" VALUES(102,'Carousel 6','ORD');
CREATE TABLE Financial_Services(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Type TEXT ,
    Location_Within_Terminal TEXT,
    Operating_Hours TEXT);
INSERT INTO "Financial_Services" VALUES(1,'First Currency','Currency Exchange Service','Near Gate 10','8 AM - 6 PM');
INSERT INTO "Financial_Services" VALUES(2,'Second Currency','Currency Exchange Service','Before Security Check','7 AM - 9 PM');
INSERT INTO "Financial_Services" VALUES(3,'Bank of America ATM','ATM Machine','Near Gate 10','Open 24 Hours');
INSERT INTO "Financial_Services" VALUES(4,'Citi Bank Branch','ATM Machine','After Security Check','8 AM - 5 PM');
CREATE TABLE Flight_Details(
    flight_number INT,
    departure_time TIMESTAMP,
    arrival_time TIMESTAMP,
    origin_airport TEXT,
    destination_airport TEXT,
    airline_id INTEGER,
    FOREIGN KEY (airline_id) REFERENCES Airlines(id));
INSERT INTO "Flight_Details" VALUES(101,'2023-04-15 06:00:00','2023-04-15 08:00:00','LAX','JFK',1);
INSERT INTO "Flight_Details" VALUES(102,'2023-04-15 07:00:00','2023-04-15 09:00:00','ORD','LHR',2);
CREATE TABLE Terminals(
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT);
INSERT INTO "Terminals" VALUES(1,'Terminal A','Gate Area');
INSERT INTO "Terminals" VALUES(2,'Terminal B','International Departure');
COMMIT;
