BEGIN TRANSACTION;
CREATE TABLE Airports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Airports" VALUES(1,'New York JFK','New York City, USA');
INSERT INTO "Airports" VALUES(2,'Los Angeles LAX','Los Angeles, USA');
INSERT INTO "Airports" VALUES(3,'London Heathrow','London, UK');
INSERT INTO "Airports" VALUES(4,'Paris CDG','Paris, France');
INSERT INTO "Airports" VALUES(5,'Tokyo Narita','Tokyo, Japan');
CREATE TABLE Architects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Architects" VALUES(1,'John Doe','john.doe@example.com');
INSERT INTO "Architects" VALUES(2,'Jane Smith','jane.smith@example.com');
INSERT INTO "Architects" VALUES(3,'Michael Johnson','michael.johnson@example.com');
INSERT INTO "Architects" VALUES(4,'Emily Davis','emily.davis@example.com');
INSERT INTO "Architects" VALUES(5,'William Brown','william.brown@example.com');
CREATE TABLE "Buildings" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Building_Design TEXT,
    airport_terminal TEXT REFERENCES Terminals(terminal_name)
);
INSERT INTO "Buildings" VALUES(1,'Studio A','Terminal 1');
INSERT INTO "Buildings" VALUES(2,'Studio B','Terminal 2');
INSERT INTO "Buildings" VALUES(3,'Studio C','Terminal 4');
INSERT INTO "Buildings" VALUES(4,'Studio D','Terminal A');
INSERT INTO "Buildings" VALUES(5,'Studio E','Terminal 2');
CREATE TABLE Flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number TEXT,
    departure_airport TEXT REFERENCES Airports(name),
    arrival_airport TEXT REFERENCES Airports(name)
);
INSERT INTO "Flights" VALUES(1,'AA123','New York JFK','London Heathrow');
INSERT INTO "Flights" VALUES(2,'DL456','Los Angeles LAX','Paris CDG');
INSERT INTO "Flights" VALUES(3,'BA789','London Heathrow','New York JFK');
INSERT INTO "Flights" VALUES(4,'AF101','Paris CDG','Tokyo Narita');
INSERT INTO "Flights" VALUES(5,'LH234','Frankfurt FRA','New York JFK');
CREATE TABLE Terminals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airport_id INTEGER REFERENCES Airports(id),
    terminal_name TEXT,
    capacity INT DEFAULT 0
);
INSERT INTO "Terminals" VALUES(1,1,'Terminal 1',20);
INSERT INTO "Terminals" VALUES(2,1,'Terminal 2',30);
INSERT INTO "Terminals" VALUES(3,2,'Terminal 4',40);
INSERT INTO "Terminals" VALUES(4,3,'Terminal 2',50);
INSERT INTO "Terminals" VALUES(5,4,'Terminal A',60);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Architects',5);
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Terminals',5);
INSERT INTO "sqlite_sequence" VALUES('Flights',5);
INSERT INTO "sqlite_sequence" VALUES('Buildings',5);
COMMIT;
