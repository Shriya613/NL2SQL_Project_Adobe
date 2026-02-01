BEGIN TRANSACTION;
CREATE TABLE Amusement_Parks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Opening_Year INTEGER,
    Attractions INT
);
INSERT INTO "Amusement_Parks" VALUES(1,'Disney World','Orlando, Florida',1971,20);
INSERT INTO "Amusement_Parks" VALUES(2,'Universal Studios','Los Angeles, California',1964,35);
CREATE TABLE Cinemas(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Genre TEXT,
    Seats INT
);
INSERT INTO "Cinemas" VALUES(1,'AMC Orlando 24','Orlando, Florida','Action',800);
INSERT INTO "Cinemas" VALUES(2,'Movie Theater of Los Angeles','Los Angeles, California','Drama',1000);
CREATE TABLE Concert_Venues(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Artist TEXT,
    Date DATE
);
INSERT INTO "Concert_Venues" VALUES(1,'Hard Rock Live','Orlando, Florida','Bruce Springsteen','June 1st');
INSERT INTO "Concert_Venues" VALUES(2,'Staples Center','Los Angeles, California','Taylor Swift','August 1st');
CREATE TABLE Restaurants(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Cuisine TYPE,
    Price_Range REAL
);
INSERT INTO "Restaurants" VALUES(1,'Le Bernardin','New York City, New York','French',150.0);
INSERT INTO "Restaurants" VALUES(2,'Girl & the Goat','Chicago, Illinois','American',100.0);
CREATE TABLE Theaters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Show_Time TIME,
    Ticket_Price REAL
);
INSERT INTO "Theaters" VALUES(1,'Broadway Playhouse','New York City, New York','7 PM',100.0);
INSERT INTO "Theaters" VALUES(2,'Downtown Theatre','Chicago, Illinois','8 PM',85.0);
CREATE TABLE Water_Parks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT,
    Opening_Month DATE,
    Pool_Type TEXT
);
INSERT INTO "Water_Parks" VALUES(1,'Typhoon Lagoon','Orlando, Florida','May','Surf Pool');
INSERT INTO "Water_Parks" VALUES(2,'Blizzard Beach','Orlando, Florida','February','Snow Park');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Amusement_Parks',2);
INSERT INTO "sqlite_sequence" VALUES('Water_Parks',2);
INSERT INTO "sqlite_sequence" VALUES('Cinemas',2);
INSERT INTO "sqlite_sequence" VALUES('Concert_Venues',2);
INSERT INTO "sqlite_sequence" VALUES('Theaters',2);
INSERT INTO "sqlite_sequence" VALUES('Restaurants',2);
COMMIT;
