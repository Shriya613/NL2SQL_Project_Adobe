BEGIN TRANSACTION;
CREATE TABLE Hosts (
    host_id INTEGER PRIMARY KEY AUTOINCREMENT,
    host_name TEXT,
    host_gender CHAR(1) 
);
INSERT INTO "Hosts" VALUES(1,'John Doe','M');
INSERT INTO "Hosts" VALUES(2,'Jane Smith','F');
INSERT INTO "Hosts" VALUES(3,'Bob Johnson','M');
INSERT INTO "Hosts" VALUES(4,'Alice Brown','F');
INSERT INTO "Hosts" VALUES(5,'Tom Clark','M');
CREATE TABLE Programs (
    program_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_type_name TEXT
);
INSERT INTO "Programs" VALUES(1,'Morning Show');
INSERT INTO "Programs" VALUES(2,'Afternoon Drive');
INSERT INTO "Programs" VALUES(3,'Evening News');
INSERT INTO "Programs" VALUES(4,'Night Music');
INSERT INTO "Programs" VALUES(5,'Weekend Special');
CREATE TABLE Radio_Stations (
    station_id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_name TEXT,
    frequency REAL
);
INSERT INTO "Radio_Stations" VALUES(1,'Classic Rock',96.3);
INSERT INTO "Radio_Stations" VALUES(2,'Top Hits',88.7);
INSERT INTO "Radio_Stations" VALUES(3,'Country Gold',92.5);
INSERT INTO "Radio_Stations" VALUES(4,'Jazz & Blues',100.1);
INSERT INTO "Radio_Stations" VALUES(5,'Electronic Beats',99.9);
CREATE TABLE Radio_Stations_Hosts (
    radio_station_id INTEGER,
    host_id INTEGER,
    FOREIGN KEY(radio_station_id) REFERENCES Radio_Stations(station_id),
    FOREIGN KEY(host_id) REFERENCES Hosts(host_id),
    PRIMARY KEY (radio_station_id, host_id)
);
INSERT INTO "Radio_Stations_Hosts" VALUES(1,1);
INSERT INTO "Radio_Stations_Hosts" VALUES(2,2);
INSERT INTO "Radio_Stations_Hosts" VALUES(3,3);
INSERT INTO "Radio_Stations_Hosts" VALUES(4,4);
INSERT INTO "Radio_Stations_Hosts" VALUES(5,5);
CREATE TABLE Radio_Stations_Programs (
    radio_station_id INTEGER,
    program_type_id INTEGER,
    day_of_week INTEGER ,
    start_time TIME ,
    end_time TIME ,
    FOREIGN KEY(radio_station_id) REFERENCES Radio_Stations(station_id),
    FOREIGN KEY(program_type_id) REFERENCES Programs(program_type_id),
    PRIMARY KEY (radio_station_id, program_type_id, day_of_week, start_time)
);
INSERT INTO "Radio_Stations_Programs" VALUES(1,1,1,'06:00:00','10:00:00');
INSERT INTO "Radio_Stations_Programs" VALUES(1,2,2,'12:00:00','16:00:00');
INSERT INTO "Radio_Stations_Programs" VALUES(2,1,1,'07:00:00','11:00:00');
INSERT INTO "Radio_Stations_Programs" VALUES(3,3,1,'06:00:00','10:00:00');
INSERT INTO "Radio_Stations_Programs" VALUES(4,2,2,'12:00:00','16:00:00');
INSERT INTO "Radio_Stations_Programs" VALUES(3,1,1,'06:00:00','10:00:00');
INSERT INTO "Radio_Stations_Programs" VALUES(4,1,2,'12:00:00','16:00:00');
INSERT INTO "Radio_Stations_Programs" VALUES(5,1,NULL,NULL,NULL);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Radio_Stations',5);
INSERT INTO "sqlite_sequence" VALUES('Programs',5);
INSERT INTO "sqlite_sequence" VALUES('Hosts',5);
COMMIT;
