BEGIN TRANSACTION;
CREATE TABLE Aircrafts(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT,
    Model TEXT,
    Capacity INT);
INSERT INTO "Aircrafts" VALUES(1,'Passenger','Boing 777',300);
INSERT INTO "Aircrafts" VALUES(2,'Cargo','MD-11',100);
INSERT INTO "Aircrafts" VALUES(3,'Private Jet','Gulfstream V',15);
CREATE TABLE Assignments(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Flight_ID INTEGER,
    Pilot_ID INTEGER,
    CoPilot_ID INTEGER,
    FOREIGN KEY(Flight_ID) REFERENCES Flights(ID),
    FOREIGN KEY(Pilot_ID) REFERENCES Pilots(ID),
    FOREIGN KEY(CoPilot_ID) REFERENCES Pilots(ID));
INSERT INTO "Assignments" VALUES(1,1,1,2);
INSERT INTO "Assignments" VALUES(2,2,3,4);
INSERT INTO "Assignments" VALUES(3,3,1,2);
INSERT INTO "Assignments" VALUES(4,4,3,4);
CREATE TABLE Flight_Attendants(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT,
    Experience INT,
    HireDate DATE,
    Training_Center_ID INTEGER,
    FOREIGN KEY(Training_Center_ID) REFERENCES Training_Center(ID));
INSERT INTO "Flight_Attendants" VALUES(1,'John Doe',28,3,'2019-06-01',1);
INSERT INTO "Flight_Attendants" VALUES(2,'Jane Smith',27,2,'2018-08-15',2);
INSERT INTO "Flight_Attendants" VALUES(3,'Michael Johnson',30,5,'2015-01-01',1);
INSERT INTO "Flight_Attendants" VALUES(4,'Emily Davis',29,4,'2017-10-10',2);
CREATE TABLE Flights(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    DepartureTime TIMESTAMP,
    ArrivalTime TIMESTAMP,
    OriginCity TEXT,
    DestinationCity TEXT,
    Aircraft_ID INTEGER,
    FOREIGN KEY(Aircraft_ID) REFERENCES Aircrafts(ID));
INSERT INTO "Flights" VALUES(1,'2023-01-01 08:00:00','2023-01-01 12:00:00','Los Angeles','Chicago',1);
INSERT INTO "Flights" VALUES(2,'2023-02-01 10:00:00','2023-02-01 14:00:00','San Francisco','Seattle',2);
INSERT INTO "Flights" VALUES(3,'2023-03-01 12:00:00','2023-03-01 16:00:00','Miami','Atlanta',1);
INSERT INTO "Flights" VALUES(4,'2023-04-01 09:00:00','2023-04-01 13:00:00','Houston','Denver',2);
CREATE TABLE Pilots(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT,
    Experience INT,
    LicenseNumber TEXT,
    Training_Center_ID INTEGER,
    FOREIGN KEY(Training_Center_ID) REFERENCES Training_Center(ID));
INSERT INTO "Pilots" VALUES(1,'David Brown',35,7,'LBAA1234C',1);
INSERT INTO "Pilots" VALUES(2,'Karen Miller',31,6,'LAAB3456D',2);
INSERT INTO "Pilots" VALUES(3,'Daniel Wilson',38,8,'LLBA5678E',1);
INSERT INTO "Pilots" VALUES(4,'Laura Taylor',32,5,'LMNA9012F',2);
CREATE TABLE Training_Center(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE);
INSERT INTO "Training_Center" VALUES(1,'Aviation Academ');
INSERT INTO "Training_Center" VALUES(2,'New Horizons Aviation');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Training_Center',2);
INSERT INTO "sqlite_sequence" VALUES('Flight_Attendants',4);
INSERT INTO "sqlite_sequence" VALUES('Pilots',4);
INSERT INTO "sqlite_sequence" VALUES('Aircrafts',3);
INSERT INTO "sqlite_sequence" VALUES('Flights',4);
INSERT INTO "sqlite_sequence" VALUES('Assignments',4);
COMMIT;
