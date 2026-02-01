BEGIN TRANSACTION;
CREATE TABLE Bar(
    barID INTEGER PRIMARY KEY AUTOINCREMENT,
    airportID INTEGER,
    name TEXT,
    location TEXT,
    capacity INTEGER ,
    openingHours TEXT
);
INSERT INTO "Bar" VALUES(1,1,'Gateway Lounge','Concourse A',50,'9 AM to 11 PM');
INSERT INTO "Bar" VALUES(2,2,'Beer Garden','Concourse B',75,'11 AM to 9 PM');
INSERT INTO "Bar" VALUES(3,3,'Whiskey Sky','Concourse C',100,'10 AM to 8 PM');
INSERT INTO "Bar" VALUES(4,4,'Rooftop Wine Bar','Concourse D',70,'8 AM to 6 PM');
INSERT INTO "Bar" VALUES(5,5,'Sunset Margarita Bar','Concourse E',80,'7 AM to 7 PM');
CREATE TABLE Employee(
    employeeID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    position TEXT,
    hireDate DATE,
    salary REAL 
);
INSERT INTO "Employee" VALUES(1,'John Doe','Pilot','2020-06-01',100000.0);
INSERT INTO "Employee" VALUES(2,'Jane Smith','Flight Attendant','2021-05-01',50000.0);
INSERT INTO "Employee" VALUES(3,'Bob Johnson','Mechanic','2019-07-01',60000.0);
INSERT INTO "Employee" VALUES(4,'Emily Davis','Customer Service Representative','2022-08-01',40000.0);
INSERT INTO "Employee" VALUES(5,'Chris Brown','Security Officer','2018-09-01',35000.0);
CREATE TABLE Flight(
    flightID INTEGER PRIMARY KEY AUTOINCREMENT,
    airline TEXT,
    departureAirport TEXT,
    arrivalAirport TEXT,
    date TIMESTAMP,
    duration NUMERIC 
);
INSERT INTO "Flight" VALUES(1,'United','EWR','IAH','2023-08-01 06:00:00',4);
INSERT INTO "Flight" VALUES(2,'Delta','ATL','LAX','2023-08-02 07:00:00',5);
INSERT INTO "Flight" VALUES(3,'American Airlines','DFW','MIA','2023-08-03 08:00:00',4);
INSERT INTO "Flight" VALUES(4,'Southwest','MCO','PHX','2023-08-04 09:00:00',3);
INSERT INTO "Flight" VALUES(5,'Alaska Airlines','SEA','PDX','2023-08-05 10:00:00',2);
CREATE TABLE Passenger(
    passengerID INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    birthdate DATE,
    nationality TEXT
);
INSERT INTO "Passenger" VALUES(1,'Alexander','Smith','1990-01-15','USA');
INSERT INTO "Passenger" VALUES(2,'Samantha','Johnson','1995-02-20','UK');
INSERT INTO "Passenger" VALUES(3,'David','Williams','1988-03-25','Canada');
INSERT INTO "Passenger" VALUES(4,'Emma','Martinez','2000-04-05','Spain');
INSERT INTO "Passenger" VALUES(5,'Michael','Davis','1992-05-10','Australia');
CREATE TABLE Restaurant(
    restaurantID INTEGER PRIMARY KEY AUTOINCREMENT,
    airportID INTEGER,
    name TEXT,
    location TEXT,
    type TEXT,
    openingHours TEXT
);
INSERT INTO "Restaurant" VALUES(1,1,'Grand Central Oyster Bar','Terminal A','Seafood','9 AM to 11 PM');
INSERT INTO "Restaurant" VALUES(2,2,'The Flying Pizza','Terminal B','Italian','11 AM to 9 PM');
INSERT INTO "Restaurant" VALUES(3,3,'Taste of Texas BBQ','Terminal C','BBQ','10 AM to 8 PM');
INSERT INTO "Restaurant" VALUES(4,4,'Café de la Maison','Terminal D','French','8 AM to 6 PM');
INSERT INTO "Restaurant" VALUES(5,5,'Green Leaf Salad Bar','Terminal E','Salad','7 AM to 7 PM');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Flight',5);
INSERT INTO "sqlite_sequence" VALUES('Employee',5);
INSERT INTO "sqlite_sequence" VALUES('Passenger',5);
INSERT INTO "sqlite_sequence" VALUES('Restaurant',5);
INSERT INTO "sqlite_sequence" VALUES('Bar',5);
COMMIT;
