BEGIN TRANSACTION;
CREATE TABLE Amenities (
    amenityId INTEGER PRIMARY KEY AUTOINCREMENT,
    amenityName TEXT);
INSERT INTO "Amenities" VALUES(1,'Spa');
INSERT INTO "Amenities" VALUES(2,'Gym');
INSERT INTO "Amenities" VALUES(3,'Pool');
INSERT INTO "Amenities" VALUES(4,'Restaurant');
INSERT INTO "Amenities" VALUES(5,'Bar');
CREATE TABLE Hotel_Amenities (
    hotelId INTEGER,
    amenityId INTEGER,
    FOREIGN KEY(hotelId) REFERENCES Hotels(id),
    FOREIGN KEY(amenityId) REFERENCES Amenities(amenityId));
INSERT INTO "Hotel_Amenities" VALUES(1,1);
INSERT INTO "Hotel_Amenities" VALUES(2,1);
INSERT INTO "Hotel_Amenities" VALUES(3,1);
INSERT INTO "Hotel_Amenities" VALUES(4,1);
INSERT INTO "Hotel_Amenities" VALUES(5,1);
INSERT INTO "Hotel_Amenities" VALUES(5,4);
CREATE TABLE Hotels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Hotels" VALUES(1,'The Ritz','London');
INSERT INTO "Hotels" VALUES(2,'Le Royal Monceau','Paris');
INSERT INTO "Hotels" VALUES(3,'Grand Hyatt','New York');
INSERT INTO "Hotels" VALUES(4,'Hilton Tokyo','Tokyo');
INSERT INTO "Hotels" VALUES(5,'Four Seasons','Dubai');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hotels',5);
INSERT INTO "sqlite_sequence" VALUES('Amenities',5);
COMMIT;
