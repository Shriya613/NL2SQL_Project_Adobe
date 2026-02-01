BEGIN TRANSACTION;
CREATE TABLE Streaming_Service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    price REAL
);
INSERT INTO "Streaming_Service" VALUES(1,'Netflix',12.99);
INSERT INTO "Streaming_Service" VALUES(2,'Amazon Prime Video',8.99);
INSERT INTO "Streaming_Service" VALUES(3,'Disney+',6.99);
INSERT INTO "Streaming_Service" VALUES(4,'Hulu',5.99);
INSERT INTO "Streaming_Service" VALUES(5,'YouTube Premium',11.99);
CREATE TABLE Streaming_Service_Video_Quality (
    streaming_service_id INTEGER,
    video_quality_id INTEGER,
    FOREIGN KEY (streaming_service_id) REFERENCES Streaming_Service(id),
    FOREIGN KEY (video_quality_id) REFERENCES Video_Quality(id),
    PRIMARY KEY (streaming_service_id, video_quality_id)
);
INSERT INTO "Streaming_Service_Video_Quality" VALUES(1,1);
INSERT INTO "Streaming_Service_Video_Quality" VALUES(2,1);
INSERT INTO "Streaming_Service_Video_Quality" VALUES(3,1);
INSERT INTO "Streaming_Service_Video_Quality" VALUES(4,1);
INSERT INTO "Streaming_Service_Video_Quality" VALUES(5,1);
INSERT INTO "Streaming_Service_Video_Quality" VALUES(5,3);
CREATE TABLE Video_Quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quality TEXT
);
INSERT INTO "Video_Quality" VALUES(1,'HD');
INSERT INTO "Video_Quality" VALUES(2,'SD');
INSERT INTO "Video_Quality" VALUES(3,'Ultra HD');
INSERT INTO "Video_Quality" VALUES(4,'FHD');
INSERT INTO "Video_Quality" VALUES(5,'3D');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Service',5);
INSERT INTO "sqlite_sequence" VALUES('Video_Quality',5);
COMMIT;
