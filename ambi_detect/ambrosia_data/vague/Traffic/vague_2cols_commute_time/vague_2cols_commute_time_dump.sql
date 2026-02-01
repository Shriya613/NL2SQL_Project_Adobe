BEGIN TRANSACTION;
CREATE TABLE Accidents (
    accident_date DATE,
    neighborhood_id INTEGER,
    num_accident INT ,
    FOREIGN KEY(neighborhood_id) REFERENCES Neighborhood(id)
);
INSERT INTO "Accidents" VALUES('2022-01-01',1,2);
INSERT INTO "Accidents" VALUES('2022-01-02',2,4);
INSERT INTO "Accidents" VALUES('2022-01-03',3,3);
INSERT INTO "Accidents" VALUES('2022-01-04',4,5);
INSERT INTO "Accidents" VALUES('2022-01-05',5,2);
CREATE TABLE DailyTrafficVolume (
    date DATE,
    neighborhood_id INTEGER,
    total_traffic INT ,
    FOREIGN KEY(neighborhood_id) REFERENCES Neighborhood(id)
);
INSERT INTO "DailyTrafficVolume" VALUES('2022-01-01',1,3000);
INSERT INTO "DailyTrafficVolume" VALUES('2022-01-02',2,4500);
INSERT INTO "DailyTrafficVolume" VALUES('2022-01-03',3,6000);
INSERT INTO "DailyTrafficVolume" VALUES('2022-01-04',4,5500);
INSERT INTO "DailyTrafficVolume" VALUES('2022-01-05',5,4200);
CREATE TABLE Neighborhood (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    population INTEGER ,
    mean_commute_time REAL ,
    median_commute_time REAL 
);
INSERT INTO "Neighborhood" VALUES(1,'Downtown',8000,30.5,27.6);
INSERT INTO "Neighborhood" VALUES(2,'Midtown',9500,28.5,25.3);
INSERT INTO "Neighborhood" VALUES(3,'Southside',11000,29.8,27.2);
INSERT INTO "Neighborhood" VALUES(4,'Northside',12000,30.2,26.9);
INSERT INTO "Neighborhood" VALUES(5,'Eastside',8500,31.5,28.4);
CREATE TABLE SpeedLimits (
    road_segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    street_name TEXT,
    neighborhood_id INTEGER,
    start_point LATITUDE,
    end_point LONGITUDE,
    max_limit INT ,
    min_limit INT ,
    FOREIGN KEY(neighborhood_id) REFERENCES Neighborhood(id)
);
INSERT INTO "SpeedLimits" VALUES(1,'Main St.',1,-79.999,-79.998,40,30);
INSERT INTO "SpeedLimits" VALUES(2,'Elm St.',2,-79.995,-79.994,30,20);
INSERT INTO "SpeedLimits" VALUES(3,'Oak St.',3,-79.993,-79.992,30,20);
INSERT INTO "SpeedLimits" VALUES(4,'Maple St.',4,-79.989,-79.988,30,20);
INSERT INTO "SpeedLimits" VALUES(5,'Pine St.',5,-79.987,-79.986,40,30);
CREATE TABLE VehicleType (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name VARCHAR(255),
    count_passing INT 
);
INSERT INTO "VehicleType" VALUES(1,'Car',500);
INSERT INTO "VehicleType" VALUES(2,'Truck',100);
INSERT INTO "VehicleType" VALUES(3,'Bicycle',50);
INSERT INTO "VehicleType" VALUES(4,'Pedestrian',200);
INSERT INTO "VehicleType" VALUES(5,'Bus',20);
CREATE TABLE WeatherConditions (
    condition_datetime DATETIME,
    neighborhood_id INTEGER,
    rainfall DECIMAL(10,2) ,
    temperature DECIMAL(10,2) ,
    windspeed MPH ,
    FOREIGN KEY(neighborhood_id) REFERENCES Neighborhood(id)
);
INSERT INTO "WeatherConditions" VALUES('2022-01-01 08:00:00',1,0.2,10.5,5);
INSERT INTO "WeatherConditions" VALUES('2022-01-01 12:00:00',1,0,12,7);
INSERT INTO "WeatherConditions" VALUES('2022-01-01 16:00:00',1,0,8,4);
INSERT INTO "WeatherConditions" VALUES('2022-01-01 20:00:00',1,0,5,2);
INSERT INTO "WeatherConditions" VALUES('2022-01-02 08:00:00',2,0.1,10.8,6);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Neighborhood',5);
INSERT INTO "sqlite_sequence" VALUES('VehicleType',5);
INSERT INTO "sqlite_sequence" VALUES('SpeedLimits',5);
COMMIT;
