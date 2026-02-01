BEGIN TRANSACTION;
CREATE TABLE Airport (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    RunwayLength REAL
);
INSERT INTO "Airport" VALUES(1,'JFK International','New York City',9876.0);
INSERT INTO "Airport" VALUES(2,'Dulles International','Washington D.C.',10345.0);
CREATE TABLE CarRental (
    ID INTEGER PRIMARY KEY,
    AirportID INTEGER,
    Airport_Name TEXT,
    Company TEXT,
    Rates TEXT,
    ContactInfo TEXT,
    FOREIGN KEY(AirportID) REFERENCES Airport(ID)
);
INSERT INTO "CarRental" VALUES(1,1,'JFK International','Avis','Moderate','+1 800-331-1212');
INSERT INTO "CarRental" VALUES(2,1,'JFK International','Enterprise','Low','+1 855-263-3522');
INSERT INTO "CarRental" VALUES(3,2,'Dulles International','National','High','+1 877-222-9350');
INSERT INTO "CarRental" VALUES(4,2,'Dulles International','Hertz','Moderate','+1 800-654-3131');
CREATE TABLE Hotel (
    ID INTEGER PRIMARY KEY,
    AirportID INTEGER,
    Airport_Name TEXT,
    Airport_Hotel TEXT,
    Price TEXT,
    FOREIGN KEY(AirportID) REFERENCES Airport(ID)
);
INSERT INTO "Hotel" VALUES(1,1,'JFK International','Hyatt Regency JFK','150');
INSERT INTO "Hotel" VALUES(2,1,'JFK International','Marriott JFK','200');
INSERT INTO "Hotel" VALUES(3,2,'Dulles International','Westin Dulles','180');
INSERT INTO "Hotel" VALUES(4,2,'Dulles International','Hilton Dulles','220');
CREATE TABLE ParkingLot (
    ID INTEGER PRIMARY KEY,
    AirportID INTEGER,
    Airport_Name TEXT,
    Type TEXT,
    Spaces INT,
    Rate PERCENT,
    FOREIGN KEY(AirportID) REFERENCES Airport(ID)
);
INSERT INTO "ParkingLot" VALUES(1,1,'JFK International','Short Term',100,30);
INSERT INTO "ParkingLot" VALUES(2,1,'JFK International','Long Term',500,20);
INSERT INTO "ParkingLot" VALUES(3,2,'Dulles International','Economy',300,15);
INSERT INTO "ParkingLot" VALUES(4,2,'Dulles International','Premium',100,35);
CREATE TABLE Restaurant (
    ID INTEGER PRIMARY KEY,
    AirportID INTEGER,
    Airport_Name TEXT,
    Name TEXT,
    Cuisine TEXT,
    FOREIGN KEY(AirportID) REFERENCES Airport(ID)
);
INSERT INTO "Restaurant" VALUES(1,1,'JFK International','LeGrand NYC','Italian');
INSERT INTO "Restaurant" VALUES(2,1,'JFK International','Burger Bar','Fast Food');
INSERT INTO "Restaurant" VALUES(3,2,'Dulles International','The Silver Diner','American');
INSERT INTO "Restaurant" VALUES(4,2,'Dulles International','Taste of India','Indian');
CREATE TABLE SleepPod (
    ID INTEGER PRIMARY KEY,
    AirportID INTEGER,
    Airport_Name TEXT,
    Sleep_Pod TEXT,
    Price TEXT,
    FOREIGN KEY(AirportID) REFERENCES Airport(ID)
);
INSERT INTO "SleepPod" VALUES(1,1,'JFK International','GoSleep Pods','150');
INSERT INTO "SleepPod" VALUES(2,1,'JFK International','YOUpod','200');
INSERT INTO "SleepPod" VALUES(3,2,'Dulles International','RestZen Capsules','180');
INSERT INTO "SleepPod" VALUES(4,2,'Dulles International','Napcab Private Suites','220');
COMMIT;
