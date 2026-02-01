BEGIN TRANSACTION;
CREATE TABLE Airlines(
    id INTEGER PRIMARY KEY,
    airline_name TEXT,
    type_of_alliance TEXT ,
    home_airport_id INTEGER,
    FOREIGN KEY(home_airport_id) REFERENCES Airports(id));
INSERT INTO "Airlines" VALUES(1,'Alliance Airways','Airline Alliance',1);
INSERT INTO "Airlines" VALUES(2,'Budget Airlines','Airline Alliance',2);
INSERT INTO "Airlines" VALUES(3,'Fly High Independent','Independent Carrier',3);
INSERT INTO "Airlines" VALUES(4,'Global Travellers','Independent Carrier',4);
INSERT INTO "Airlines" VALUES(5,'Oceanic Flyers','No Alliance',5);
CREATE TABLE Airports(
    id INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT,
    country TEXT,
    latitude REAL,
    longitude REAL);
INSERT INTO "Airports" VALUES(1,'John F Kennedy International','New York City','USA',40.641987,-73.778604);
INSERT INTO "Airports" VALUES(2,'Los Angeles International','Los Angeles','USA',33.944826,-118.405889);
INSERT INTO "Airports" VALUES(3,'Heathrow Airport','London','UK',51.477025,-0.461866);
INSERT INTO "Airports" VALUES(4,'Dubai International','Dubai','UAE',25.259648,55.285619);
INSERT INTO "Airports" VALUES(5,'Tokyo Haneda','Tokyo','Japan',35.601194,139.737522);
CREATE TABLE Assignments(
    assignment_id INTEGER PRIMARY KEY,
    flight_number INTEGER,
    pilot_license_number INTEGER,
    FOREIGN KEY(flight_number) REFERENCES Flights(flight_number),
    FOREIGN KEY(pilot_license_number) REFERENCES Pilots(pilot_license_number) );
INSERT INTO "Assignments" VALUES(1,1,1);
INSERT INTO "Assignments" VALUES(2,1,2);
INSERT INTO "Assignments" VALUES(3,2,3);
INSERT INTO "Assignments" VALUES(4,2,4);
INSERT INTO "Assignments" VALUES(5,3,1);
CREATE TABLE Flights(
    flight_number INTEGER PRIMARY KEY,
    departure_time DATETIME,
    arrival_time DATETIME,
    airline_id INTEGER,
    aircraft_model TEXT,
    source_airport_id INTEGER,
    destination_airport_id INTEGER,
    FOREIGN KEY(airline_id) REFERENCES Airlines(id),
    FOREIGN KEY(source_airport_id) REFERENCES Airports(id),
    FOREIGN KEY(destination_airport_id) REFERENCES Airports(id));
INSERT INTO "Flights" VALUES(1,'2022-01-01 10:00:00','2022-01-01 14:00:00',1,'Airbus A320',1,2);
INSERT INTO "Flights" VALUES(2,'2022-01-01 12:00:00','2022-01-01 17:00:00',2,'Boeing B737',3,4);
INSERT INTO "Flights" VALUES(3,'2022-01-02 08:00:00','2022-01-02 12:00:00',3,'Airbus A320',4,5);
INSERT INTO "Flights" VALUES(4,'2022-01-02 10:00:00','2022-01-02 15:00:00',4,'Boeing B787',5,1);
INSERT INTO "Flights" VALUES(5,'2022-01-03 11:00:00','2022-01-03 16:00:00',5,'Airbus A320',2,3);
CREATE TABLE Pilots(
    pilot_license_number INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    rating_level TEXT  );
INSERT INTO "Pilots" VALUES(1,'John','Smith','1990-01-01','A');
INSERT INTO "Pilots" VALUES(2,'Jane','Doe','1985-05-05','A');
INSERT INTO "Pilots" VALUES(3,'Alice','Wright','1989-02-12','B');
INSERT INTO "Pilots" VALUES(4,'Bob','Johnson','1992-03-13','C');
COMMIT;
