BEGIN TRANSACTION;
CREATE TABLE Air_traffic_controller(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Experience INT ,
    Salary TEXT,
    Training_Center TEXT
);
INSERT INTO "Air_traffic_controller" VALUES(1,'John Doe',8,'60000 per year','ATC Academy');
INSERT INTO "Air_traffic_controller" VALUES(2,'Jane Smith',12,'70000 per year','Aviation Institute');
CREATE TABLE Flight(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Date DATE,
    Time TIME,
    Destination TEXT,
    Pilot_id INTEGER,
    FOREIGN KEY (Pilot_id) REFERENCES Pilots(id)
);
INSERT INTO "Flight" VALUES(1,'2022-01-01','10:00:00','New York',1);
INSERT INTO "Flight" VALUES(2,'2022-01-02','15:00:00','Los Angeles',2);
CREATE TABLE Ground_staff(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Experience INT ,
    Salary TEXT,
    Role TEXT
);
INSERT INTO "Ground_staff" VALUES(1,'Mike Johnson',5,'60000 per year','Baggage Handler');
INSERT INTO "Ground_staff" VALUES(2,'Emma Brown',3,'55000 per year','Gate Agent');
CREATE TABLE Pilots(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    License_Number TEXT,
    Total_Flights INT 
);
INSERT INTO "Pilots" VALUES(1,'Tom Williams','ABCD1234',300);
INSERT INTO "Pilots" VALUES(2,'Linda Davis','EFGH5678',500);
CREATE TABLE Planes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Model TEXT,
    Registration_number TEXT,
    Capacity INT ,
    Current_Location INTEGER,
    FOREIGN KEY (Current_Location) REFERENCES Runway(id)
);
INSERT INTO "Planes" VALUES(1,'Boeing 737','N12345',200,1);
INSERT INTO "Planes" VALUES(2,'Airbus A320','N67890',220,2);
CREATE TABLE Runway(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Length REAL,
    Width REAL,
    Location TEXT
);
INSERT INTO "Runway" VALUES(1,9000.0,150.0,'Chicago O''Hare International Airport');
INSERT INTO "Runway" VALUES(2,8000.0,140.0,'Dallas/Fort Worth International Airport');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Air_traffic_controller',2);
INSERT INTO "sqlite_sequence" VALUES('Ground_staff',2);
INSERT INTO "sqlite_sequence" VALUES('Flight',2);
INSERT INTO "sqlite_sequence" VALUES('Pilots',2);
INSERT INTO "sqlite_sequence" VALUES('Runway',2);
INSERT INTO "sqlite_sequence" VALUES('Planes',2);
COMMIT;
