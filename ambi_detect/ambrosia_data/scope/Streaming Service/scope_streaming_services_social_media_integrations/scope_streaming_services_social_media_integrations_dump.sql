BEGIN TRANSACTION;
CREATE TABLE Arbitrary_Table1(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column1 TEXT,
    column2 INTEGER
);
INSERT INTO "Arbitrary_Table1" VALUES(1,'Row1 Col1',10);
INSERT INTO "Arbitrary_Table1" VALUES(2,'Row2 Col1',20);
INSERT INTO "Arbitrary_Table1" VALUES(3,'Row3 Col1',30);
INSERT INTO "Arbitrary_Table1" VALUES(4,'Row4 Col1',40);
INSERT INTO "Arbitrary_Table1" VALUES(5,'Row5 Col1',50);
CREATE TABLE Arbitrary_Table2(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column1 TEXT,
    column2 INTEGER
);
INSERT INTO "Arbitrary_Table2" VALUES(1,'Row1 Col1',10);
INSERT INTO "Arbitrary_Table2" VALUES(2,'Row2 Col1',20);
INSERT INTO "Arbitrary_Table2" VALUES(3,'Row3 Col1',30);
INSERT INTO "Arbitrary_Table2" VALUES(4,'Row4 Col1',40);
INSERT INTO "Arbitrary_Table2" VALUES(5,'Row5 Col1',50);
CREATE TABLE Social_Media_Integrations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    integration_type TEXT,
    description TEXT
);
INSERT INTO "Social_Media_Integrations" VALUES(1,'Share Button','Allows users to share content');
INSERT INTO "Social_Media_Integrations" VALUES(2,'Like Button','Allows users to like content');
INSERT INTO "Social_Media_Integrations" VALUES(3,'Comment Section','Enables user comments on content');
INSERT INTO "Social_Media_Integrations" VALUES(4,'Follow Feature','Lets users follow their favorite channels');
INSERT INTO "Social_Media_Integrations" VALUES(5,'Watch Later List','A feature that lets users save videos to watch later');
CREATE TABLE Streaming_Services(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT,
    subscription_fee REAL 
);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix',7.99);
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video',8.99);
INSERT INTO "Streaming_Services" VALUES(3,'Hulu',5.99);
INSERT INTO "Streaming_Services" VALUES(4,'Disney+',6.99);
INSERT INTO "Streaming_Services" VALUES(5,'HBO Max',9.99);
CREATE TABLE Streaming_Services_Social_Media_Integrations(
    streaming_service_id INTEGER,
    social_media_integration_id INTEGER,
    FOREIGN KEY(streaming_service_id) REFERENCES Streaming_Services(id),
    FOREIGN KEY(social_media_integration_id) REFERENCES Social_Media_Integrations(id)
);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(1,1);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(1,2);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(2,3);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(3,4);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(4,5);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(2,1);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(3,1);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(4,1);
INSERT INTO "Streaming_Services_Social_Media_Integrations" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
INSERT INTO "sqlite_sequence" VALUES('Social_Media_Integrations',5);
INSERT INTO "sqlite_sequence" VALUES('Arbitrary_Table1',5);
INSERT INTO "sqlite_sequence" VALUES('Arbitrary_Table2',5);
COMMIT;
