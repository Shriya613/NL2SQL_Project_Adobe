BEGIN TRANSACTION;
CREATE TABLE Boutique_Hotels(
    hotel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Boutique_Hotels" VALUES(1,'Grand Art Hotel','Downtown');
INSERT INTO "Boutique_Hotels" VALUES(2,'Eco Lodge','Suburbs');
INSERT INTO "Boutique_Hotels" VALUES(3,'Crystal Palace','Beachside');
INSERT INTO "Boutique_Hotels" VALUES(4,'Mountain Retreat','Uptown');
INSERT INTO "Boutique_Hotels" VALUES(5,'River View Inn','City Center');
CREATE TABLE Boutique_Hotels_Themes(
    boutique_hotel_id INTEGER REFERENCES Boutique_Hotels(hotel_id) ON DELETE CASCADE,
    theme_id INTEGER REFERENCES Themes(theme_id) ON DELETE CASCADE);
INSERT INTO "Boutique_Hotels_Themes" VALUES(1,1);
INSERT INTO "Boutique_Hotels_Themes" VALUES(2,1);
INSERT INTO "Boutique_Hotels_Themes" VALUES(3,1);
INSERT INTO "Boutique_Hotels_Themes" VALUES(4,1);
INSERT INTO "Boutique_Hotels_Themes" VALUES(5,1);
INSERT INTO "Boutique_Hotels_Themes" VALUES(5,4);
CREATE TABLE Themes(
    theme_id INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_name TEXT);
INSERT INTO "Themes" VALUES(1,'Art_themed');
INSERT INTO "Themes" VALUES(2,'Historical_themed');
INSERT INTO "Themes" VALUES(3,'Nature_themed');
INSERT INTO "Themes" VALUES(4,'Modern_themed');
INSERT INTO "Themes" VALUES(5,'Romantic_themed');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Boutique_Hotels',5);
INSERT INTO "sqlite_sequence" VALUES('Themes',5);
COMMIT;
