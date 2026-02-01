BEGIN TRANSACTION;
CREATE TABLE Airlines (
    Airline_Code CHAR(2) PRIMARY KEY UNIQUE ,
    Airline_Name TEXT,
    Headquarters_City TEXT,
    Website TEXT
);
INSERT INTO "Airlines" VALUES('AA','American Airlines','Fort Worth','https://www.aa.com');
INSERT INTO "Airlines" VALUES('DL','Delta Air Lines','Atlanta','https://www.delta.com');
INSERT INTO "Airlines" VALUES('UA','United Airlines','Chicago','https://www.united.com');
INSERT INTO "Airlines" VALUES('BA','British Airways','London','https://www.britishairways.com');
INSERT INTO "Airlines" VALUES('LU','Luftansa','Frankfurt','https://www.luftansa.com');
CREATE TABLE "Airports" (
    Airport CHAR(5) PRIMARY KEY UNIQUE ,
    Country TEXT,
    City TEXT,
    Elevation INT DEFAULT 0,
    Timezone TEXT,
    IATA_Code VARCHAR(4),
    Latitude DECIMAL(9,6),
    Longitude DECIMAL(11,8)
);
INSERT INTO "Airports" VALUES('JFK','USA','New York',3,'-05:00','JFK',40.6418,-73.7789);
INSERT INTO "Airports" VALUES('LAX','USA','Los Angeles',239,'-08:00','LAX',33.9451,-118.4052);
INSERT INTO "Airports" VALUES('YYZ','Canada','Toronto',253,'-05:00','YYZ',55.6815,-79.4536);
INSERT INTO "Airports" VALUES('DUB','Ireland','Dublin',75,'GMT+00:00','DUB',53.3749,-6.1432);
INSERT INTO "Airports" VALUES('NRT','Japan','Tokyo',13,'JST+09:00','NRT',35.6051,139.7691);
CREATE TABLE Flights (
    Flight_ID INTEGER PRIMARY KEY,
    Departure_Time TIMESTAMP,
    Arrival_Time TIMESTAMP,
    Aircraft_Type TEXT,
    Origin_Airport_Code VARCHAR(5) REFERENCES Airports(Airport),
    Destination_Airport_Code VARCHAR(5) REFERENCES Airports(Airport)
);
CREATE TABLE Maintenance_Records (
    Record_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Plane_ID INTEGER REFERENCES Planes(Plane_ID),
    Work_Order_Description TEXT,
    Start_DateTime TIMESTAMP,
    End_DateTime TIMESTAMP,
    Total_Hours_Worked FLOAT,
    Cost FLOAT
);
INSERT INTO "Maintenance_Records" VALUES(1,1,'Engine overhaul','2021-01-01 08:00:00','2021-01-02 18:00:00',12.0,15000.0);
INSERT INTO "Maintenance_Records" VALUES(2,2,'Landing gear repair','2021-02-01 08:00:00','2021-02-02 18:00:00',10.0,10000.0);
INSERT INTO "Maintenance_Records" VALUES(3,3,'Interior cleaning and refurbishment','2021-03-01 08:00:00','2021-03-05 18:00:00',40.0,8000.0);
INSERT INTO "Maintenance_Records" VALUES(4,4,'Avionics upgrade','2021-04-01 08:00:00','2021-04-03 18:00:00',24.0,20000.0);
INSERT INTO "Maintenance_Records" VALUES(5,5,'Wing inspection','2021-05-01 08:00:00','2021-05-02 18:00:00',8.0,5000.0);
CREATE TABLE Pilots (
    Pilot_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    First_Name TEXT,
    Last_Name TEXT,
    Date_Of_Birth DATE,
    License_Number VARCHAR(20),
    License_Issue_Date DATE,
    Rating TEXT,
    Airline_Code CHAR(2) REFERENCES Airlines(Airline_Code)
);
INSERT INTO "Pilots" VALUES(1,'John','Smith','1990-01-01','123ABCD','2015-01-01','ATP','AA');
INSERT INTO "Pilots" VALUES(2,'Jane','Doe','1992-02-02','XYZWEE','2016-02-02','ATP','UA');
INSERT INTO "Pilots" VALUES(3,'Michael','Jackson','1991-03-03','ABCDEF','2017-03-03','ATP','DL');
INSERT INTO "Pilots" VALUES(4,'Emma','Swift','1993-04-04','GIHJKL','2018-04-04','ATP','BA');
INSERT INTO "Pilots" VALUES(5,'Paul','McCartney','1942-06-06','LMNOPQ','1964-06-06','ATP','LU');
CREATE TABLE Planes (
    Plane_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Registration_Number VARCHAR(7),
    Model_Name TEXT,
    Manufacturer_Name TEXT,
    Year_Manufactured INT,
    Capacity INT,
    Pilot_ID INTEGER REFERENCES Pilots(Pilot_ID)
);
INSERT INTO "Planes" VALUES(1,'N12345','Boeing 737-800','Boeing',2010,200,1);
INSERT INTO "Planes" VALUES(2,'EI-DVB','Airbus A320neo','Airbus',2015,180,2);
INSERT INTO "Planes" VALUES(3,'VY-BCR','Embraer E190','Embraer',2008,100,3);
INSERT INTO "Planes" VALUES(4,'SE-RML','Bombardier CRJ900','Bombardier',2006,88,4);
INSERT INTO "Planes" VALUES(5,'EC-IXQ','Airbus A380-800','Airbus',2012,500,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Pilots',5);
INSERT INTO "sqlite_sequence" VALUES('Planes',5);
INSERT INTO "sqlite_sequence" VALUES('Maintenance_Records',5);
COMMIT;
