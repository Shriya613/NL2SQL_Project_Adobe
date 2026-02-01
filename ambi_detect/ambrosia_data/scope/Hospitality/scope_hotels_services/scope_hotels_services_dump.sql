BEGIN TRANSACTION;
CREATE TABLE Hotels (
    HotelID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT
);
INSERT INTO "Hotels" VALUES(1,'Grand Plaza','Downtown');
INSERT INTO "Hotels" VALUES(2,'Park View Inn','Suburb');
INSERT INTO "Hotels" VALUES(3,'Coastal Paradise','Beachside');
INSERT INTO "Hotels" VALUES(4,'Mountain Retreat','Mountains');
INSERT INTO "Hotels" VALUES(5,'City Center Suites','Central Business District');
CREATE TABLE Hotels_Services (
    Hotels_ID INTEGER,
    Service_ID INTEGER,
    FOREIGN KEY(Hotels_ID) REFERENCES Hotels(HotelID),
    FOREIGN KEY(Service_ID) REFERENCES Services(ServiceID),
    PRIMARY KEY(Hotels_ID, Service_ID)
);
INSERT INTO "Hotels_Services" VALUES(1,2);
INSERT INTO "Hotels_Services" VALUES(1,3);
INSERT INTO "Hotels_Services" VALUES(1,4);
INSERT INTO "Hotels_Services" VALUES(1,5);
INSERT INTO "Hotels_Services" VALUES(2,1);
INSERT INTO "Hotels_Services" VALUES(2,3);
INSERT INTO "Hotels_Services" VALUES(2,4);
INSERT INTO "Hotels_Services" VALUES(2,5);
INSERT INTO "Hotels_Services" VALUES(3,1);
INSERT INTO "Hotels_Services" VALUES(3,2);
INSERT INTO "Hotels_Services" VALUES(3,4);
INSERT INTO "Hotels_Services" VALUES(3,5);
INSERT INTO "Hotels_Services" VALUES(4,1);
INSERT INTO "Hotels_Services" VALUES(4,2);
INSERT INTO "Hotels_Services" VALUES(4,3);
INSERT INTO "Hotels_Services" VALUES(4,5);
INSERT INTO "Hotels_Services" VALUES(5,1);
INSERT INTO "Hotels_Services" VALUES(5,2);
INSERT INTO "Hotels_Services" VALUES(5,3);
INSERT INTO "Hotels_Services" VALUES(5,4);
INSERT INTO "Hotels_Services" VALUES(5,5);
INSERT INTO "Hotels_Services" VALUES(1,1);
INSERT INTO "Hotels_Services" VALUES(2,2);
INSERT INTO "Hotels_Services" VALUES(3,3);
INSERT INTO "Hotels_Services" VALUES(4,4);
CREATE TABLE Services (
    ServiceID INTEGER PRIMARY KEY AUTOINCREMENT,
    ServiceName TEXT
);
INSERT INTO "Services" VALUES(1,'Housekeeping');
INSERT INTO "Services" VALUES(2,'Restaurant');
INSERT INTO "Services" VALUES(3,'Gym');
INSERT INTO "Services" VALUES(4,'Concierge');
INSERT INTO "Services" VALUES(5,'Laundry');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hotels',5);
INSERT INTO "sqlite_sequence" VALUES('Services',5);
COMMIT;
