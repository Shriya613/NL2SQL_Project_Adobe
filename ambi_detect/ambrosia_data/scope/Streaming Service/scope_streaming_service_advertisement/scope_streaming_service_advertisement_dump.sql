BEGIN TRANSACTION;
CREATE TABLE Advertisement (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    duration REAL
);
INSERT INTO "Advertisement" VALUES(1,'Pre_roll_Ads',30.0);
INSERT INTO "Advertisement" VALUES(2,'Mid_roll_Ads',60.0);
INSERT INTO "Advertisement" VALUES(3,'Post_roll_Ads',120.0);
INSERT INTO "Advertisement" VALUES(4,'Banner_Ads',10.0);
INSERT INTO "Advertisement" VALUES(5,'Interstitial_Ads',90.0);
CREATE TABLE Streaming_Service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    platform TEXT
);
INSERT INTO "Streaming_Service" VALUES(1,'Netflix','Web');
INSERT INTO "Streaming_Service" VALUES(2,'Hulu','Web');
INSERT INTO "Streaming_Service" VALUES(3,'Amazon Prime Video','Web');
INSERT INTO "Streaming_Service" VALUES(4,'Disney+','App');
INSERT INTO "Streaming_Service" VALUES(5,'HBO Max','App');
CREATE TABLE Streaming_Service_Advertisement (
    streaming_service_id INTEGER,
    advertisement_id INTEGER,
    FOREIGN KEY(streaming_service_id) REFERENCES Streaming_Service(id),
    FOREIGN KEY(advertisement_id) REFERENCES Advertisement(id)
);
INSERT INTO "Streaming_Service_Advertisement" VALUES(1,1);
INSERT INTO "Streaming_Service_Advertisement" VALUES(2,1);
INSERT INTO "Streaming_Service_Advertisement" VALUES(3,1);
INSERT INTO "Streaming_Service_Advertisement" VALUES(4,1);
INSERT INTO "Streaming_Service_Advertisement" VALUES(5,1);
INSERT INTO "Streaming_Service_Advertisement" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Service',5);
INSERT INTO "sqlite_sequence" VALUES('Advertisement',5);
COMMIT;
