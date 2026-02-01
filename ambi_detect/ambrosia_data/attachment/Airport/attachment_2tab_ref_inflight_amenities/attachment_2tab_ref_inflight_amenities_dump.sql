BEGIN TRANSACTION;
CREATE TABLE Aircraft (
    id INTEGER PRIMARY KEY,
    Name TEXT UNIQUE 
);
INSERT INTO "Aircraft" VALUES(1,'Boeing 787 Dreamliner');
INSERT INTO "Aircraft" VALUES(2,'Airbus A380');
CREATE TABLE Baggage (
    id INTEGER PRIMARY KEY,
    Weight_kg INTEGER,
    Type TEXT,
    Flight_number INTEGER,
    FOREIGN KEY (Flight_number) REFERENCES Flights (id)
);
INSERT INTO "Baggage" VALUES(1,25,'Checked Luggage',101);
INSERT INTO "Baggage" VALUES(2,10,'Carry-on Luggage',102);
INSERT INTO "Baggage" VALUES(3,15,'Checked Luggage',103);
INSERT INTO "Baggage" VALUES(4,5,'Carry-on Luggage',104);
INSERT INTO "Baggage" VALUES(5,20,'Checked Luggage',105);
CREATE TABLE Flights (
    id INTEGER PRIMARY KEY,
    Departure_time DATETIME,
    Arrival_time DATETIME,
    Origin_airport TEXT,
    Destination_airport TEXT,
    Aircraft_id INTEGER,
    FOREIGN KEY (Aircraft_id) REFERENCES Aircraft (id)
);
INSERT INTO "Flights" VALUES(101,'2022-11-01 10:00','2022-11-01 14:00','LAX','ORD',1);
INSERT INTO "Flights" VALUES(102,'2022-11-02 09:30','2022-11-02 13:30','DFW','YUL',1);
INSERT INTO "Flights" VALUES(103,'2022-11-03 10:00','2022-11-03 14:00','LAX','JFK',2);
INSERT INTO "Flights" VALUES(104,'2022-11-04 09:30','2022-11-04 13:30','DFW','YYZ',2);
INSERT INTO "Flights" VALUES(105,'2022-11-05 10:00','2022-11-05 14:00','LAX','LHR',2);
CREATE TABLE Inflight_Entertainments (
    id INTEGER PRIMARY KEY,
    Entertainment_name TEXT UNIQUE ,
    Aircraft_id INTEGER,
    FOREIGN KEY (Aircraft_id) REFERENCES Aircraft (id)
);
INSERT INTO "Inflight_Entertainments" VALUES(1,'Movies',1);
INSERT INTO "Inflight_Entertainments" VALUES(2,'Music',2);
INSERT INTO "Inflight_Entertainments" VALUES(3,'Games',2);
CREATE TABLE Onboard_Services (
    id INTEGER PRIMARY KEY,
    Service_name TEXT UNIQUE ,
    Aircraft_id INTEGER,
    FOREIGN KEY (Aircraft_id) REFERENCES Aircraft (id)
);
INSERT INTO "Onboard_Services" VALUES(1,'Free WiFi',1);
INSERT INTO "Onboard_Services" VALUES(2,'Meals',2);
INSERT INTO "Onboard_Services" VALUES(3,'In-flight magazines',2);
CREATE TABLE Passengers (
    id INTEGER PRIMARY KEY,
    First_name TEXT,
    Last_name TEXT,
    Age INTEGER,
    Flight_number INTEGER,
    FOREIGN KEY (Flight_number) REFERENCES Flights (id)
);
INSERT INTO "Passengers" VALUES(1,'John','Doe',35,101);
INSERT INTO "Passengers" VALUES(2,'Jane','Smith',28,102);
INSERT INTO "Passengers" VALUES(3,'Alice','Johnson',42,103);
INSERT INTO "Passengers" VALUES(4,'Bob','Williams',60,104);
INSERT INTO "Passengers" VALUES(5,'Emma','Martin',22,105);
COMMIT;
