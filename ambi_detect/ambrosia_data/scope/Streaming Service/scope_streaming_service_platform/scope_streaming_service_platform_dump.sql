BEGIN TRANSACTION;
CREATE TABLE Platform (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Platform" VALUES(1,'Web_Browser');
INSERT INTO "Platform" VALUES(2,'Mobile_App');
INSERT INTO "Platform" VALUES(3,'Smart_TV');
INSERT INTO "Platform" VALUES(4,'Gaming_Console');
INSERT INTO "Platform" VALUES(5,'Desktop_Application');
CREATE TABLE Streaming_Service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT,
    description TEXT
);
INSERT INTO "Streaming_Service" VALUES(1,'Netflix','Worldwide popular video-on-demand service.');
INSERT INTO "Streaming_Service" VALUES(2,'Amazon_Prime','Video-on-demand service by Amazon.');
INSERT INTO "Streaming_Service" VALUES(3,'Hulu','American subscription video on demand service.');
INSERT INTO "Streaming_Service" VALUES(4,'Disney_Plus','Subscription video-on-demand over-the-top content from Disney.');
INSERT INTO "Streaming_Service" VALUES(5,'YouTube','Free video sharing website that allows users to upload, view, rate, share videos.');
CREATE TABLE Streaming_Service_Platform (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streaming_id INTEGER,
    platform_id INTEGER,
    FOREIGN KEY (streaming_id) REFERENCES Streaming_Service(id),
    FOREIGN KEY (platform_id) REFERENCES Platform(id)
);
INSERT INTO "Streaming_Service_Platform" VALUES(1,1,1);
INSERT INTO "Streaming_Service_Platform" VALUES(2,2,1);
INSERT INTO "Streaming_Service_Platform" VALUES(3,3,1);
INSERT INTO "Streaming_Service_Platform" VALUES(4,4,1);
INSERT INTO "Streaming_Service_Platform" VALUES(5,5,1);
INSERT INTO "Streaming_Service_Platform" VALUES(6,5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Platform',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Service',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Service_Platform',6);
COMMIT;
