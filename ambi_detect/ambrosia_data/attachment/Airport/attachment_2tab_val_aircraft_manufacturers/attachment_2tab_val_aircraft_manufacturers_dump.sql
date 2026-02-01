BEGIN TRANSACTION;
CREATE TABLE Airbus(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Model TEXT,
    Number_of_seats INT,
    Manufacture_year INT,
    LastMaintenanceDate DATE
);
INSERT INTO "Airbus" VALUES(1,'A320',180,2010,'2022-07-01');
INSERT INTO "Airbus" VALUES(2,'A380',800,2007,'2021-12-01');
CREATE TABLE Boeing(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Model TEXT,
    Number_of_seats INT,
    Manufacture_year INT,
    LastMaintenanceDate DATE
);
INSERT INTO "Boeing" VALUES(1,'737',180,2000,'2022-06-01');
INSERT INTO "Boeing" VALUES(2,'747',400,1990,'2021-09-01');
CREATE TABLE Flight(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airplaneId INTEGER,
    departureTime TIMESTAMP,
    arrivalTime TIMESTAMP,
    destination TEXT,
    FOREIGN KEY (airplaneId) REFERENCES Airbus(id),
    FOREIGN KEY (airplaneId) REFERENCES Boeing(id)
);
INSERT INTO "Flight" VALUES(1,1,'2022-08-01 08:00:00','2022-08-01 12:00:00','New York');
INSERT INTO "Flight" VALUES(2,2,'2022-08-02 09:00:00','2022-08-02 13:00:00','Los Angeles');
CREATE TABLE Maintenance(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airplaneId INTEGER,
    maintenanceType TEXT,
    datePerformed DATE,
    technicianName TEXT,
    FOREIGN KEY (airplaneId) REFERENCES Airbus(id),
    FOREIGN KEY (airplaneId) REFERENCES Boeing(id)
);
INSERT INTO "Maintenance" VALUES(1,1,'Engine Overhaul','2021-05-01','Mike Brown');
INSERT INTO "Maintenance" VALUES(2,2,'Landing Gear Repair','2021-06-01','Lisa Johnson');
CREATE TABLE Pilot(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    licenseNumber TEXT,
    experienceYears INT 
);
INSERT INTO "Pilot" VALUES(1,'John Doe','ABC1234',10);
INSERT INTO "Pilot" VALUES(2,'Jane Smith','XYZ7890',15);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Boeing',2);
INSERT INTO "sqlite_sequence" VALUES('Airbus',2);
INSERT INTO "sqlite_sequence" VALUES('Flight',2);
INSERT INTO "sqlite_sequence" VALUES('Pilot',2);
INSERT INTO "sqlite_sequence" VALUES('Maintenance',2);
COMMIT;
