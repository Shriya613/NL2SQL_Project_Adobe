BEGIN TRANSACTION;
CREATE TABLE DeviceUsers(
    deviceId INTEGER,
    userId INTEGER,
    FOREIGN KEY (deviceId) REFERENCES Devices_Compatibility(id),
    FOREIGN KEY (userId) REFERENCES Users(id));
CREATE TABLE Devices_Compatibility(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deviceName TEXT,
    isActive BOOLEAN);
INSERT INTO "Devices_Compatibility" VALUES(1,'Smart_TV',1);
INSERT INTO "Devices_Compatibility" VALUES(2,'Laptop',1);
INSERT INTO "Devices_Compatibility" VALUES(3,'Mobile Phone',1);
INSERT INTO "Devices_Compatibility" VALUES(4,'Tablet',1);
INSERT INTO "Devices_Compatibility" VALUES(5,'Gaming Console',0);
CREATE TABLE Streaming_Service_Devices_Compatibility(
    streamingServiceId INTEGER,
    deviceTypeId INTEGER,
    FOREIGN KEY (streamingServiceId) REFERENCES Streaming_Services(id),
    FOREIGN KEY (deviceTypeId) REFERENCES Devices_Compatibility(id));
INSERT INTO "Streaming_Service_Devices_Compatibility" VALUES(1,1);
INSERT INTO "Streaming_Service_Devices_Compatibility" VALUES(2,1);
INSERT INTO "Streaming_Service_Devices_Compatibility" VALUES(3,1);
INSERT INTO "Streaming_Service_Devices_Compatibility" VALUES(4,1);
INSERT INTO "Streaming_Service_Devices_Compatibility" VALUES(5,1);
INSERT INTO "Streaming_Service_Devices_Compatibility" VALUES(5,2);
CREATE TABLE Streaming_Services(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    serviceName TEXT,
    subscriptionPrice REAL);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix',8.99);
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video',12.99);
INSERT INTO "Streaming_Services" VALUES(3,'Hulu',6.99);
INSERT INTO "Streaming_Services" VALUES(4,'Disney+',7.99);
INSERT INTO "Streaming_Services" VALUES(5,'Apple TV+',4.99);
CREATE TABLE SubscriptionPlans(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    planName TEXT,
    price PERSISTENT);
INSERT INTO "SubscriptionPlans" VALUES(1,'Basic Plan',5);
INSERT INTO "SubscriptionPlans" VALUES(2,'Standard Plan',10);
INSERT INTO "SubscriptionPlans" VALUES(3,'Premium Plan',15);
INSERT INTO "SubscriptionPlans" VALUES(4,'Ultra Plan',20);
CREATE TABLE UserSubscriptions(
    userId INTEGER,
    streamingServiceId INTEGER,
    subscriptionPlanId INTEGER,
    FOREIGN KEY (userId) REFERENCES Users(id),
    FOREIGN KEY (streamingServiceId) REFERENCES Streaming_Services(id),
    FOREIGN KEY (subscriptionPlanId) REFERENCES SubscriptionPlans(id));
CREATE TABLE Users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userName TEXT,
    emailAddress TEXT);
INSERT INTO "Users" VALUES(1,'JohnDoe','johndoe@example.com');
INSERT INTO "Users" VALUES(2,'JaneDoe','janedoe@example.com');
INSERT INTO "Users" VALUES(3,'MikeSmith','mikesmith@example.com');
INSERT INTO "Users" VALUES(4,'AliceBrown','alicebrown@example.com');
INSERT INTO "Users" VALUES(5,'BobWhite','bobwhite@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Devices_Compatibility',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
INSERT INTO "sqlite_sequence" VALUES('SubscriptionPlans',4);
INSERT INTO "sqlite_sequence" VALUES('Users',5);
COMMIT;
