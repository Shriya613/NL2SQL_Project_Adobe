BEGIN TRANSACTION;
CREATE TABLE AirportCity(
    airportId INTEGER,
    cityId INTEGER,
    FOREIGN KEY(airportId) REFERENCES Airports(airportId),
    FOREIGN KEY(cityId) REFERENCES Cities(cityId)
);
INSERT INTO "AirportCity" VALUES(1,1);
INSERT INTO "AirportCity" VALUES(2,2);
INSERT INTO "AirportCity" VALUES(3,3);
INSERT INTO "AirportCity" VALUES(4,4);
INSERT INTO "AirportCity" VALUES(5,5);
CREATE TABLE Airports(
    airportId INTEGER PRIMARY KEY,
    cityId INTEGER,
    name TEXT,
    country TEXT
);
INSERT INTO "Airports" VALUES(1,1,'JFK','USA');
INSERT INTO "Airports" VALUES(2,2,'LAX','USA');
INSERT INTO "Airports" VALUES(3,3,'LHR','UK');
INSERT INTO "Airports" VALUES(4,4,'CDG','France');
INSERT INTO "Airports" VALUES(5,5,'AUH','UAE');
CREATE TABLE "Boarding_Areas"(
    areaId INTEGER PRIMARY KEY,
    location TEXT,
    capacity INTEGER
);
INSERT INTO "Boarding_Areas" VALUES(1,'Area A',10000);
INSERT INTO "Boarding_Areas" VALUES(2,'Area B',12000);
INSERT INTO "Boarding_Areas" VALUES(3,'Area C',11000);
INSERT INTO "Boarding_Areas" VALUES(4,'Area D',13000);
INSERT INTO "Boarding_Areas" VALUES(5,'Area E',10000);
CREATE TABLE Boarding_Areas_Gates(
    areaId INTEGER,
    gateId INTEGER,
    FOREIGN KEY(areaId) REFERENCES Boarding_Areas(areaId),
    FOREIGN KEY(gateId) REFERENCES Gates(gateId)
);
INSERT INTO "Boarding_Areas_Gates" VALUES(1,1);
INSERT INTO "Boarding_Areas_Gates" VALUES(1,2);
INSERT INTO "Boarding_Areas_Gates" VALUES(2,3);
INSERT INTO "Boarding_Areas_Gates" VALUES(2,4);
INSERT INTO "Boarding_Areas_Gates" VALUES(3,5);
INSERT INTO "Boarding_Areas_Gates" VALUES(2,1);
INSERT INTO "Boarding_Areas_Gates" VALUES(3,1);
INSERT INTO "Boarding_Areas_Gates" VALUES(4,1);
INSERT INTO "Boarding_Areas_Gates" VALUES(5,1);
CREATE TABLE Cities(
    cityId INTEGER PRIMARY KEY,
    name TEXT,
    population INTEGER
);
INSERT INTO "Cities" VALUES(1,'New York City',8600000);
INSERT INTO "Cities" VALUES(2,'Los Angeles',3970000);
INSERT INTO "Cities" VALUES(3,'London',9400000);
INSERT INTO "Cities" VALUES(4,'Paris',2140000);
INSERT INTO "Cities" VALUES(5,'Abu Dhabi',1500000);
CREATE TABLE "Gates"(
    gateId INTEGER PRIMARY KEY,
    gateName TEXT
);
INSERT INTO "Gates" VALUES(1,'Gate_A1');
INSERT INTO "Gates" VALUES(2,'Gate_B2');
INSERT INTO "Gates" VALUES(3,'Gate_C3');
INSERT INTO "Gates" VALUES(4,'Gate_D4');
INSERT INTO "Gates" VALUES(5,'Gate_E5');
COMMIT;
