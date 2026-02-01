BEGIN TRANSACTION;
CREATE TABLE Accessibility_Features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    featureName TEXT);
INSERT INTO "Accessibility_Features" VALUES(1,'Closed_Captioning');
INSERT INTO "Accessibility_Features" VALUES(2,'Audio_Description');
INSERT INTO "Accessibility_Features" VALUES(3,'Subtitles');
INSERT INTO "Accessibility_Features" VALUES(4,'Sign_Language_Support');
INSERT INTO "Accessibility_Features" VALUES(5,'Adaptive_Audio');
CREATE TABLE Streaming_Services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    serviceName TEXT,
    basePrice REAL);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix',12.99);
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video',8.99);
INSERT INTO "Streaming_Services" VALUES(3,'Hulu',7.99);
INSERT INTO "Streaming_Services" VALUES(4,'Disney+',6.99);
INSERT INTO "Streaming_Services" VALUES(5,'HBO Max',14.99);
CREATE TABLE Streaming_Services_Accessibility_Features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streamingServiceId INTEGER,
    accessibilityFeatureId INTEGER,
    FOREIGN KEY(streamingServiceId) REFERENCES Streaming_Services(id),
    FOREIGN KEY(accessibilityFeatureId) REFERENCES Accessibility_Features(id));
INSERT INTO "Streaming_Services_Accessibility_Features" VALUES(1,1,1);
INSERT INTO "Streaming_Services_Accessibility_Features" VALUES(2,2,1);
INSERT INTO "Streaming_Services_Accessibility_Features" VALUES(3,3,1);
INSERT INTO "Streaming_Services_Accessibility_Features" VALUES(4,4,1);
INSERT INTO "Streaming_Services_Accessibility_Features" VALUES(5,5,1);
INSERT INTO "Streaming_Services_Accessibility_Features" VALUES(6,5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Accessibility_Features',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services_Accessibility_Features',6);
COMMIT;
