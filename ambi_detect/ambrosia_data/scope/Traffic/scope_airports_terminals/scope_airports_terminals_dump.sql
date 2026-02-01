BEGIN TRANSACTION;
CREATE TABLE Airports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Airports" VALUES(1,'New York Airport','New York City');
INSERT INTO "Airports" VALUES(2,'Los Angeles Airport','California');
INSERT INTO "Airports" VALUES(3,'London Heathrow','England');
INSERT INTO "Airports" VALUES(4,'Tokyo Haneda','Japan');
INSERT INTO "Airports" VALUES(5,'Dubai International','United Arab Emirates');
CREATE TABLE Airports_Terminals (
    airport_id INTEGER,
    terminal_id INTEGER,
    FOREIGN KEY (airport_id) REFERENCES Airports(id),
    FOREIGN KEY (terminal_id) REFERENCES Terminals(id)
);
INSERT INTO "Airports_Terminals" VALUES(1,1);
INSERT INTO "Airports_Terminals" VALUES(1,2);
INSERT INTO "Airports_Terminals" VALUES(2,3);
INSERT INTO "Airports_Terminals" VALUES(3,4);
INSERT INTO "Airports_Terminals" VALUES(4,5);
INSERT INTO "Airports_Terminals" VALUES(2,1);
INSERT INTO "Airports_Terminals" VALUES(3,1);
INSERT INTO "Airports_Terminals" VALUES(4,1);
INSERT INTO "Airports_Terminals" VALUES(5,1);
CREATE TABLE Terminals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT ,
    airport_id INTEGER,
    FOREIGN KEY (airport_id) REFERENCES Airports(id)
);
INSERT INTO "Terminals" VALUES(1,'International_Terminal',1);
INSERT INTO "Terminals" VALUES(2,'Domestic_Terminal',2);
INSERT INTO "Terminals" VALUES(3,'Cargo_Terminal',3);
INSERT INTO "Terminals" VALUES(4,'General_Aviation_Terminal',4);
INSERT INTO "Terminals" VALUES(5,'Private_Passenger_Terminal',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Airports',5);
INSERT INTO "sqlite_sequence" VALUES('Terminals',5);
COMMIT;
