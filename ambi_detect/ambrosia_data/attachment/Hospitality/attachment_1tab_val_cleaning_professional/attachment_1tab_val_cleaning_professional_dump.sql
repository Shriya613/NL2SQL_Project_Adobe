BEGIN TRANSACTION;
CREATE TABLE Cleaning_Professionals(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Position TEXT,
    Hotel_ID INTEGER
);
INSERT INTO "Cleaning_Professionals" VALUES(1,'John Doe','Housekeeper',1);
INSERT INTO "Cleaning_Professionals" VALUES(2,'Jane Smith','Housekeeper',2);
INSERT INTO "Cleaning_Professionals" VALUES(3,'Mike Johnson','Laundry Attendant',1);
INSERT INTO "Cleaning_Professionals" VALUES(4,'Emily Davis','Laundry Attendant',2);
INSERT INTO "Cleaning_Professionals" VALUES(5,'William Brown','Room Attendant',3);
INSERT INTO "Cleaning_Professionals" VALUES(6,'Anna Wilson','Porter',4);
INSERT INTO "Cleaning_Professionals" VALUES(7,'David Miller','Manager',1);
INSERT INTO "Cleaning_Professionals" VALUES(8,'Sarah Jones','Assistant Manager',2);
CREATE TABLE Hotels(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Manager_Name TEXT
);
INSERT INTO "Hotels" VALUES(1,'The Grand Hotel','Downtown','David Miller');
INSERT INTO "Hotels" VALUES(2,'City Center Inn','Uptown','Sarah Jones');
INSERT INTO "Hotels" VALUES(3,'Waterfront View','Seaside',NULL);
INSERT INTO "Hotels" VALUES(4,'Cozy Inn','Suburbia',NULL);
CREATE TABLE Rooms(
    ID INTEGER PRIMARY KEY,
    Number TEXT,
    Type TEXT,
    Floor INTEGER,
    Hotel_ID INTEGER
);
INSERT INTO "Rooms" VALUES(1,'101','Deluxe Suite',1,1);
INSERT INTO "Rooms" VALUES(2,'202','Standard Room',2,1);
INSERT INTO "Rooms" VALUES(3,'303','Family Suite',3,2);
INSERT INTO "Rooms" VALUES(4,'404','Executive Room',4,3);
INSERT INTO "Rooms" VALUES(5,'505','Superior Room',5,4);
INSERT INTO "Rooms" VALUES(6,'606','Budget Room',6,1);
INSERT INTO "Rooms" VALUES(7,'707','Standard Suite',7,2);
INSERT INTO "Rooms" VALUES(8,'808','Economy Room',8,3);
CREATE TABLE Services(
    ID INTEGER PRIMARY KEY,
    Service_Type TEXT,
    Description TEXT,
    Price REAL
);
INSERT INTO "Services" VALUES(1,'Accommodation','Comfortable stay at our hotel.',99.99);
INSERT INTO "Services" VALUES(2,'Breakfast','Full English breakfast included.',15.0);
INSERT INTO "Services" VALUES(3,'Transport','Shuttle service available.',20.0);
INSERT INTO "Services" VALUES(4,'Entertainment','Live music on weekends.',10.0);
INSERT INTO "Services" VALUES(5,'Fitness Centre','State of the art gymnasium.',5.0);
COMMIT;
