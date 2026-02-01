BEGIN TRANSACTION;
CREATE TABLE Privacy_Settings (
    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    setting_value TEXT
);
INSERT INTO "Privacy_Settings" VALUES(1,'Data Collection Opt-Out');
INSERT INTO "Privacy_Settings" VALUES(2,'Content Recommendations');
INSERT INTO "Privacy_Settings" VALUES(3,'Advertising Preferences');
INSERT INTO "Privacy_Settings" VALUES(4,'Viewing History');
INSERT INTO "Privacy_Settings" VALUES(5,'Profile Information Sharing');
CREATE TABLE Streaming_Services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT,
    subscription_price REAL
);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix',12.99);
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video',8.99);
INSERT INTO "Streaming_Services" VALUES(3,'Disney+',7.99);
INSERT INTO "Streaming_Services" VALUES(4,'Hulu',6.99);
INSERT INTO "Streaming_Services" VALUES(5,'HBO Max',14.99);
CREATE TABLE Streaming_Services_Privacy_Settings (
    streaming_service_id INTEGER,
    privacy_settings_id INTEGER,
    FOREIGN KEY(streaming_service_id) REFERENCES Streaming_Services(id),
    FOREIGN KEY(privacy_settings_id) REFERENCES Privacy_Settings(setting_id)
);
INSERT INTO "Streaming_Services_Privacy_Settings" VALUES(1,1);
INSERT INTO "Streaming_Services_Privacy_Settings" VALUES(2,1);
INSERT INTO "Streaming_Services_Privacy_Settings" VALUES(3,1);
INSERT INTO "Streaming_Services_Privacy_Settings" VALUES(4,1);
INSERT INTO "Streaming_Services_Privacy_Settings" VALUES(5,1);
INSERT INTO "Streaming_Services_Privacy_Settings" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
INSERT INTO "sqlite_sequence" VALUES('Privacy_Settings',5);
COMMIT;
