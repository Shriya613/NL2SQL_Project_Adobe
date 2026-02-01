BEGIN TRANSACTION;
CREATE TABLE Aircraft(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Model VARCHAR(50),
    Capacity INTEGER,
    Manufacturer VARCHAR(50),
    Year INT);
INSERT INTO "Aircraft" VALUES(1,'Boeing 737 MAX',230,'Boeing',2022);
INSERT INTO "Aircraft" VALUES(2,'Airbus A320neo',180,'Airbus',2021);
INSERT INTO "Aircraft" VALUES(3,'Embraer E195-E2',120,'Embraer',2019);
INSERT INTO "Aircraft" VALUES(4,'Bombardier CRJ900',90,'Bombardier',2018);
INSERT INTO "Aircraft" VALUES(5,'De Havilland Canada DHC-8 Q400',78,'De Havilland Canada',2017);
CREATE TABLE Development_Projects(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ProjectName TEXT,
    Increased_Capacity TEXT,
    StartDate DATE,
    EndDate DATE,
    Cost REAL);
INSERT INTO "Development_Projects" VALUES(1,'Airport Expansion','1000 Passengers per Day','2023-01-01','2023-12-31',1000000.0);
INSERT INTO "Development_Projects" VALUES(2,'Airport Expansion','500 Passengers per Day','2024-01-01','2024-12-31',800000.0);
INSERT INTO "Development_Projects" VALUES(3,'Renovation','1000 Passengers per Day','2023-07-01','2023-12-31',600000.0);
INSERT INTO "Development_Projects" VALUES(4,'Renovation','500 Passengers per Day','2024-01-01','2024-12-31',500000.0);
INSERT INTO "Development_Projects" VALUES(5,'Terminal Upgrade','200 Passengers per Day','2023-09-01','2023-12-31',400000.0);
CREATE TABLE Flight(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureTime TIME,
    ArrivalTime TIME,
    Origin VARCHAR(50),
    Destination VARCHAR(50),
    AircraftID INTEGER,
    FOREIGN KEY (AircraftID) REFERENCES Aircraft(ID));
INSERT INTO "Flight" VALUES(1,'07:00 AM','08:30 AM','New York','Los Angeles',1);
INSERT INTO "Flight" VALUES(2,'09:00 AM','10:30 AM','Chicago','Miami',2);
INSERT INTO "Flight" VALUES(3,'11:00 AM','12:30 PM','San Francisco','Honolulu',3);
INSERT INTO "Flight" VALUES(4,'01:00 PM','03:30 PM','Dallas','Atlanta',4);
INSERT INTO "Flight" VALUES(5,'03:00 PM','05:30 PM','Philadelphia','Orlando',5);
CREATE TABLE Staff(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name VARCHAR(50),
    Position VARCHAR(50),
    Salary REAL,
    HireDate DATE);
INSERT INTO "Staff" VALUES(1,'John Doe','Pilot',75000.0,'2021-01-01');
INSERT INTO "Staff" VALUES(2,'Jane Smith','Flight Attendant',50000.0,'2020-07-01');
INSERT INTO "Staff" VALUES(3,'Tom Johnson','Mechanic',55000.0,'2019-01-01');
INSERT INTO "Staff" VALUES(4,'Emily Brown','Air Traffic Controller',60000.0,'2018-01-01');
INSERT INTO "Staff" VALUES(5,'William Davis','Security Officer',40000.0,'2017-01-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Development_Projects',5);
INSERT INTO "sqlite_sequence" VALUES('Aircraft',5);
INSERT INTO "sqlite_sequence" VALUES('Flight',5);
INSERT INTO "sqlite_sequence" VALUES('Staff',5);
COMMIT;
