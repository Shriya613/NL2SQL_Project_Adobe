BEGIN TRANSACTION;
CREATE TABLE Amenities(
    ID INTEGER PRIMARY KEY,
    Description TEXT);
INSERT INTO "Amenities" VALUES(1,'Free WiFi');
INSERT INTO "Amenities" VALUES(2,'Gym');
CREATE TABLE City(
    ID INTEGER PRIMARY KEY,
    Name TEXT UNIQUE);
INSERT INTO "City" VALUES(1,'Miami');
INSERT INTO "City" VALUES(2,'Los Angeles');
CREATE TABLE HotTub(
    ID INTEGER PRIMARY KEY,
    Capacity INTEGER ,
    HeatingMethod TEXT,
    FOREIGN KEY(ID) REFERENCES Hotel(ID));
INSERT INTO "HotTub" VALUES(1,5,'Electric');
INSERT INTO "HotTub" VALUES(2,8,'Gas');
CREATE TABLE Hotel(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    PhoneNumber TEXT,
    PriceRange REAL ,
    FOREIGN KEY(ID) REFERENCES City(ID));
INSERT INTO "Hotel" VALUES(1,'Palm Beach','200 Palm St.','+1234567890',150.0);
INSERT INTO "Hotel" VALUES(2,'Sunrise Inn','300 Sun Rise Ave.','+1345678901',100.0);
CREATE TABLE HotelAmenities(
    HotelID INTEGER,
    AmenityID INTEGER,
    Available BOOLEAN DEFAULT FALSE,
    FOREIGN KEY(HotelID) REFERENCES Hotel(ID),
    FOREIGN KEY(AmenityID) REFERENCES Amenities(ID));
INSERT INTO "HotelAmenities" VALUES(1,1,1);
INSERT INTO "HotelAmenities" VALUES(2,2,0);
CREATE TABLE SwimmingPool(
    ID INTEGER PRIMARY KEY,
    PoolType TEXT,
    Size TEXT,
    MaintenanceFrequency TEXT,
    FOREIGN KEY(ID) REFERENCES Hotel(ID));
INSERT INTO "SwimmingPool" VALUES(1,'Outdoor','Medium','Daily');
INSERT INTO "SwimmingPool" VALUES(2,'Indoor','Small','Weekly');
COMMIT;
