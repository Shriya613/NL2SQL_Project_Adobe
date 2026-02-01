BEGIN TRANSACTION;
CREATE TABLE Service(
    ServiceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Description TEXT,
    Price REAL);
INSERT INTO "Service" VALUES(1,'Cruise Booking',1000.0);
INSERT INTO "Service" VALUES(2,'Hotel Reservation',200.0);
INSERT INTO "Service" VALUES(3,'Flight Ticket Booking',300.0);
INSERT INTO "Service" VALUES(4,'Visa Assistance',150.0);
INSERT INTO "Service" VALUES(5,'Guided Tour',400.0);
CREATE TABLE Travel_Agency(
    AgencyID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Travel_Agency" VALUES(1,'Airport Agency','New York');
INSERT INTO "Travel_Agency" VALUES(2,'Global Travels','London');
INSERT INTO "Travel_Agency" VALUES(3,'European Adventures','Paris');
INSERT INTO "Travel_Agency" VALUES(4,'Asia Pacific Tours','Tokyo');
INSERT INTO "Travel_Agency" VALUES(5,'North American Expeditions','Los Angeles');
CREATE TABLE Travel_Agency_Service(
    AgencyID INTEGER,
    ServiceID INTEGER,
    FOREIGN KEY(AgencyID) REFERENCES Travel_Agency(AgencyID),
    FOREIGN KEY(ServiceID) REFERENCES Service(ServiceID));
INSERT INTO "Travel_Agency_Service" VALUES(1,1);
INSERT INTO "Travel_Agency_Service" VALUES(1,2);
INSERT INTO "Travel_Agency_Service" VALUES(2,1);
INSERT INTO "Travel_Agency_Service" VALUES(2,3);
INSERT INTO "Travel_Agency_Service" VALUES(3,1);
INSERT INTO "Travel_Agency_Service" VALUES(3,4);
INSERT INTO "Travel_Agency_Service" VALUES(4,1);
INSERT INTO "Travel_Agency_Service" VALUES(4,2);
INSERT INTO "Travel_Agency_Service" VALUES(5,1);
INSERT INTO "Travel_Agency_Service" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Travel_Agency',5);
INSERT INTO "sqlite_sequence" VALUES('Service',5);
COMMIT;
