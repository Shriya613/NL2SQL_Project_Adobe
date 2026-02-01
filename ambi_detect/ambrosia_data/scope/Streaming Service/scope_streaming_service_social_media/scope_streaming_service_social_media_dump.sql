BEGIN TRANSACTION;
CREATE TABLE Social_Media (
    social_media_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT 
);
INSERT INTO "Social_Media" VALUES(1,'Facebook_Share');
INSERT INTO "Social_Media" VALUES(2,'Twitter_Share');
INSERT INTO "Social_Media" VALUES(3,'Instagram_Post');
INSERT INTO "Social_Media" VALUES(4,'Pinterest_Pin');
INSERT INTO "Social_Media" VALUES(5,'LinkedIn_Update');
CREATE TABLE Streaming_Service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT,
    service_url TEXT
);
INSERT INTO "Streaming_Service" VALUES(1,'Netflix','https://www.netflix.com/');
INSERT INTO "Streaming_Service" VALUES(2,'Hulu','https://www.hulu.com/');
INSERT INTO "Streaming_Service" VALUES(3,'Amazon Prime Video','https://www.amazon.com/Prime-Video/b?ie=UTF8&node=2673912011');
INSERT INTO "Streaming_Service" VALUES(4,'Disney+','https://www.disneyplus.com/');
INSERT INTO "Streaming_Service" VALUES(5,'Apple TV+','https://tv.apple.com/');
CREATE TABLE Streaming_Service_Social_Media (
    streaming_service_id INTEGER,
    social_media_id INTEGER,
    FOREIGN KEY (streaming_service_id) REFERENCES Streaming_Service(id),
    FOREIGN KEY (social_media_id) REFERENCES Social_Media(social_media_id)
);
INSERT INTO "Streaming_Service_Social_Media" VALUES(1,1);
INSERT INTO "Streaming_Service_Social_Media" VALUES(2,1);
INSERT INTO "Streaming_Service_Social_Media" VALUES(3,1);
INSERT INTO "Streaming_Service_Social_Media" VALUES(4,1);
INSERT INTO "Streaming_Service_Social_Media" VALUES(5,1);
INSERT INTO "Streaming_Service_Social_Media" VALUES(5,4);
CREATE TABLE Streaming_Service_Subscription_Plan (
    streaming_service_id INTEGER,
    plan_id INTEGER,
    FOREIGN KEY (streaming_service_id) REFERENCES Streaming_Service(id),
    FOREIGN KEY (plan_id) REFERENCES Subscription_Plan(plan_id)
);
CREATE TABLE Subscription_Plan (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    price DECIMAL(10,2) 
);
INSERT INTO "Subscription_Plan" VALUES(1,'Basic Plan',7.99);
INSERT INTO "Subscription_Plan" VALUES(2,'Standard Plan',13.99);
INSERT INTO "Subscription_Plan" VALUES(3,'Premium Plan',19.99);
INSERT INTO "Subscription_Plan" VALUES(4,'Ultimate Plan',29.99);
INSERT INTO "Subscription_Plan" VALUES(5,'Family Plan',4.99);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Social_Media',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Service',5);
INSERT INTO "sqlite_sequence" VALUES('Subscription_Plan',5);
COMMIT;
