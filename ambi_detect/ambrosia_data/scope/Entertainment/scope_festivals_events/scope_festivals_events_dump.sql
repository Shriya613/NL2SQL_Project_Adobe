BEGIN TRANSACTION;
CREATE TABLE Events(
    EventTypeID INTEGER PRIMARY KEY AUTOINCREMENT,
    EventName TEXT,
    Description TEXT);
INSERT INTO "Events" VALUES(1,'Parade','A colorful procession featuring floats, music and dancers.');
INSERT INTO "Events" VALUES(2,'Fireworks Show','An impressive display of fireworks celebrating the festival.');
INSERT INTO "Events" VALUES(4,'Street Performances','Actors, musicians, acrobats and magicians entertain crowds on the streets.');
INSERT INTO "Events" VALUES(5,'Art Exhibitions','Exhibits showcasing traditional and contemporary art from various cultures.');
CREATE TABLE Festivals(
    FestivalID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Festivals" VALUES(1,'Carnival','Rio de Janeiro');
INSERT INTO "Festivals" VALUES(2,'Mardi Gras','New Orleans');
INSERT INTO "Festivals" VALUES(3,'Oktoberfest','Munich');
INSERT INTO "Festivals" VALUES(4,'Diwali','Varanasi');
INSERT INTO "Festivals" VALUES(5,'Tomatina','Buenoaia y Foca');
CREATE TABLE Festivals_Events(
    FestivalID INTEGER,
    EventTypeID INTEGER,
    FOREIGN KEY (FestivalID) REFERENCES Festivals(FestivalID),
    FOREIGN KEY (EventTypeID) REFERENCES Events(EventTypeID));
INSERT INTO "Festivals_Events" VALUES(1,1);
INSERT INTO "Festivals_Events" VALUES(2,1);
INSERT INTO "Festivals_Events" VALUES(3,1);
INSERT INTO "Festivals_Events" VALUES(4,1);
INSERT INTO "Festivals_Events" VALUES(5,1);
INSERT INTO "Festivals_Events" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Festivals',5);
INSERT INTO "sqlite_sequence" VALUES('Events',5);
COMMIT;
