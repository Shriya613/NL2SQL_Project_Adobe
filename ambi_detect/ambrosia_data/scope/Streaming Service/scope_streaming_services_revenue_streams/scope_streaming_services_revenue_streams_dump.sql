BEGIN TRANSACTION;
CREATE TABLE Revenue_Streams(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Revenue_Streams" VALUES(1,'Advertising_Revenue','This is a revenue stream from advertising');
INSERT INTO "Revenue_Streams" VALUES(2,'Subscriptions','This is a revenue stream from user subscriptions');
INSERT INTO "Revenue_Streams" VALUES(3,'Premium Content Sales','This is a revenue stream from selling premium content');
INSERT INTO "Revenue_Streams" VALUES(4,'Merchandise Sales','This is a revenue stream from sales of branded merchandise');
INSERT INTO "Revenue_Streams" VALUES(5,'Pay-Per-View Events','This is a revenue stream from pay-per-view events');
CREATE TABLE Streaming_Service_Revenue(
    service_id INTEGER,
    stream_id INTEGER,
    FOREIGN KEY (service_id) REFERENCES Streaming_Services(id),
    FOREIGN KEY (stream_id) REFERENCES Revenue_Streams(id),
    PRIMARY KEY (service_id, stream_id)
);
INSERT INTO "Streaming_Service_Revenue" VALUES(1,1);
INSERT INTO "Streaming_Service_Revenue" VALUES(1,2);
INSERT INTO "Streaming_Service_Revenue" VALUES(2,1);
INSERT INTO "Streaming_Service_Revenue" VALUES(2,3);
INSERT INTO "Streaming_Service_Revenue" VALUES(3,1);
INSERT INTO "Streaming_Service_Revenue" VALUES(4,1);
INSERT INTO "Streaming_Service_Revenue" VALUES(5,1);
CREATE TABLE Streaming_Services(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    launch_date DATE
);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix','2013-08-29');
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video','2016-03-25');
INSERT INTO "Streaming_Services" VALUES(3,'Hulu','2007-10-29');
INSERT INTO "Streaming_Services" VALUES(4,'Disney+','2019-11-12');
INSERT INTO "Streaming_Services" VALUES(5,'Apple TV+','2019-11-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Revenue_Streams',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
COMMIT;
