BEGIN TRANSACTION;
CREATE TABLE Activities(
    ID INTEGER PRIMARY KEY,
    ActivityName TEXT UNIQUE);
INSERT INTO "Activities" VALUES(1,'The Eiffel Tower');
INSERT INTO "Activities" VALUES(2,'Colosseum');
INSERT INTO "Activities" VALUES(3,'Statue of Liberty');
INSERT INTO "Activities" VALUES(4,'Great Wall of China');
INSERT INTO "Activities" VALUES(5,'Christ the Redeemer');
CREATE TABLE Locations(
    ID INTEGER PRIMARY KEY,
    LocationName TEXT UNIQUE);
INSERT INTO "Locations" VALUES(1,'Paris');
INSERT INTO "Locations" VALUES(2,'Rome');
INSERT INTO "Locations" VALUES(3,'New York City');
INSERT INTO "Locations" VALUES(4,'Beijing');
INSERT INTO "Locations" VALUES(5,'Rio de Janeiro');
CREATE TABLE Markets(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Activity TEXT,
    Location TEXT);
INSERT INTO "Markets" VALUES(1,'Central Food Market','Local Market','Portobello');
INSERT INTO "Markets" VALUES(2,'Downtown Groceries','Local Market','Somewhere Else');
INSERT INTO "Markets" VALUES(3,'Harbor Bazaar','Local Market','Another Place');
INSERT INTO "Markets" VALUES(4,'Uptown Fare','Local Market','Yet Another Place');
INSERT INTO "Markets" VALUES(5,'Riverfront Pantry','Local Market','A Different Place');
CREATE TABLE Tourists(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Activity TEXT,
    Location TEXT);
INSERT INTO "Tourists" VALUES(1,'John Doe','Tourist Attraction','Portobello');
INSERT INTO "Tourists" VALUES(2,'Jane Smith','Tourist Attraction','Somewhere Else');
INSERT INTO "Tourists" VALUES(3,'Alice Johnson','Local Market','Portobello');
INSERT INTO "Tourists" VALUES(4,'Bob Brown','Local Market','Somewhere Else');
INSERT INTO "Tourists" VALUES(5,'Charlie Davis','Tourist Attraction','Another Place');
CREATE TABLE VisitRecords(
    ID INTEGER PRIMARY KEY,
    VisitorID INTEGER,
    TouristAttractionID INTEGER,
    LocalMarketID INTEGER,
    FOREIGN KEY (VisitorID) REFERENCES Visitors(ID),
    FOREIGN KEY (TouristAttractionID) REFERENCES Tourists(ID),
    FOREIGN KEY (LocalMarketID) REFERENCES Markets(ID));
INSERT INTO "VisitRecords" VALUES(1,1,1,1);
INSERT INTO "VisitRecords" VALUES(2,1,2,2);
INSERT INTO "VisitRecords" VALUES(3,2,3,3);
INSERT INTO "VisitRecords" VALUES(4,2,4,4);
INSERT INTO "VisitRecords" VALUES(5,3,5,5);
CREATE TABLE Visitors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    DateOfVisit DATETIME);
INSERT INTO "Visitors" VALUES(1,'Michael Thompson','2023-01-01 10:00:00');
INSERT INTO "Visitors" VALUES(2,'Emily Watson','2023-01-02 12:00:00');
INSERT INTO "Visitors" VALUES(3,'David Jackson','2023-01-03 14:00:00');
INSERT INTO "Visitors" VALUES(4,'Sophia King','2023-01-04 16:00:00');
INSERT INTO "Visitors" VALUES(5,'Liam Scott','2023-01-05 18:00:00');
COMMIT;
