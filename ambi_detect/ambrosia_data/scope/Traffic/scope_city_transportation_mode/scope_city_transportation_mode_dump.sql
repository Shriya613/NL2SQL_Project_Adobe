BEGIN TRANSACTION;
CREATE TABLE Accidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accident_type TEXT,
    severity TEXT,
    road_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (road_id) REFERENCES Roads(id) ON DELETE CASCADE
);
INSERT INTO "Accidents" VALUES(1,'Fender Bender','Minor',1,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(2,'Hit and Run','Moderate',2,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(3,'Head-on Collision','Major',3,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(4,'Multi Vehicle Pileup','Catastrophic',4,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(5,'DUI Accident','Serious',5,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(6,'Distracted Driver Accident','Moderate',6,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(7,'Speeding Ticket','Minor',7,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(8,'Reckless Driving','Major',8,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(9,'Texting While Driving','Minor',9,'2024-01-30 18:39:55');
INSERT INTO "Accidents" VALUES(10,'Red Light Running','Serious',10,'2024-01-30 18:39:55');
CREATE TABLE City (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    population INTEGER
);
INSERT INTO "City" VALUES(1,'New York',8419000);
INSERT INTO "City" VALUES(2,'Los Angeles',3792600);
INSERT INTO "City" VALUES(3,'Chicago',2693910);
INSERT INTO "City" VALUES(4,'Houston',2209945);
INSERT INTO "City" VALUES(5,'Phoenix',1660273);
INSERT INTO "City" VALUES(6,'Philadelphia',1526000);
CREATE TABLE City_Transportation_Mode (
    city_id INTEGER,
    transportation_mode_id INTEGER,
    FOREIGN KEY (city_id) REFERENCES City(id) ON DELETE CASCADE,
    FOREIGN KEY (transportation_mode_id) REFERENCES Transportation_Mode(id) ON DELETE CASCADE,
    PRIMARY KEY (city_id, transportation_mode_id)
);
INSERT INTO "City_Transportation_Mode" VALUES(1,1);
INSERT INTO "City_Transportation_Mode" VALUES(1,2);
INSERT INTO "City_Transportation_Mode" VALUES(2,1);
INSERT INTO "City_Transportation_Mode" VALUES(2,2);
INSERT INTO "City_Transportation_Mode" VALUES(3,1);
INSERT INTO "City_Transportation_Mode" VALUES(3,3);
INSERT INTO "City_Transportation_Mode" VALUES(4,1);
INSERT INTO "City_Transportation_Mode" VALUES(4,4);
INSERT INTO "City_Transportation_Mode" VALUES(5,1);
INSERT INTO "City_Transportation_Mode" VALUES(5,5);
INSERT INTO "City_Transportation_Mode" VALUES(6,1);
CREATE TABLE Roads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    road_id INTEGER,
    city_id INTEGER,
    length REAL,
    FOREIGN KEY (city_id) REFERENCES City(id) ON DELETE CASCADE
);
INSERT INTO "Roads" VALUES(1,1,1,10.0);
INSERT INTO "Roads" VALUES(2,2,1,15.0);
INSERT INTO "Roads" VALUES(3,3,2,20.0);
INSERT INTO "Roads" VALUES(4,4,2,25.0);
INSERT INTO "Roads" VALUES(5,5,3,30.0);
INSERT INTO "Roads" VALUES(6,6,3,35.0);
INSERT INTO "Roads" VALUES(7,7,4,40.0);
INSERT INTO "Roads" VALUES(8,8,4,45.0);
INSERT INTO "Roads" VALUES(9,9,5,50.0);
INSERT INTO "Roads" VALUES(10,10,5,55.0);
CREATE TABLE Traffic_Volume (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    volume INTEGER,
    road_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (road_id) REFERENCES Roads(id) ON DELETE SET NULL
);
INSERT INTO "Traffic_Volume" VALUES(1,1000,1,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(2,1500,2,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(3,2000,3,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(4,2500,4,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(5,3000,5,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(6,3500,6,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(7,4000,7,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(8,4500,8,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(9,5000,9,'2024-01-30 18:39:55');
INSERT INTO "Traffic_Volume" VALUES(10,5500,10,'2024-01-30 18:39:55');
CREATE TABLE Transportation_Mode (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT
);
INSERT INTO "Transportation_Mode" VALUES(1,'Buses');
INSERT INTO "Transportation_Mode" VALUES(2,'Trains');
INSERT INTO "Transportation_Mode" VALUES(3,'Taxis');
INSERT INTO "Transportation_Mode" VALUES(4,'Cycling');
INSERT INTO "Transportation_Mode" VALUES(5,'Walking');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('City',6);
INSERT INTO "sqlite_sequence" VALUES('Transportation_Mode',5);
INSERT INTO "sqlite_sequence" VALUES('Roads',10);
INSERT INTO "sqlite_sequence" VALUES('Traffic_Volume',10);
INSERT INTO "sqlite_sequence" VALUES('Accidents',10);
COMMIT;
