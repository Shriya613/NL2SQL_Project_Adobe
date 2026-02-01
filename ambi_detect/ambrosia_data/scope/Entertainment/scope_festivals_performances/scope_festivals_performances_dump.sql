BEGIN TRANSACTION;
CREATE TABLE Artists(
    ArtistID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Genre TEXT);
INSERT INTO "Artists" VALUES(1,'The Beatles','Rock');
INSERT INTO "Artists" VALUES(2,'David Bowie','Rock');
INSERT INTO "Artists" VALUES(3,'Ellie Goulding','Pop');
INSERT INTO "Artists" VALUES(4,'Katy Perry','Pop');
INSERT INTO "Artists" VALUES(5,'Michael Jackson','Pop');
CREATE TABLE Festivals(
    FestivalID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Festivals" VALUES(1,'Summerfest','Milwaukee');
INSERT INTO "Festivals" VALUES(2,'Coachella Valley Music And Arts Festival','Indio, California');
INSERT INTO "Festivals" VALUES(3,'Glastonbury Festival','Pilton, England');
INSERT INTO "Festivals" VALUES(4,'Burning Man','Black Rock Desert, Nevada');
INSERT INTO "Festivals" VALUES(5,'Electric Daisy Carnival','Las Vegas');
CREATE TABLE Festivals_Artists(
    FestivalID INTEGER,
    ArtistID INTEGER,
    FOREIGN KEY(FestivalID) REFERENCES Festivals(FestivalID),
    FOREIGN KEY(ArtistID) REFERENCES Artists(ArtistID));
CREATE TABLE Festivals_Performances(
    FestivalID INTEGER,
    PerformanceID INTEGER,
    Date DATETIME,
    FOREIGN KEY(FestivalID) REFERENCES Festivals(FestivalID),
    FOREIGN KEY(PerformanceID) REFERENCES Performances(PerformanceID));
INSERT INTO "Festivals_Performances" VALUES(1,1,NULL);
INSERT INTO "Festivals_Performances" VALUES(2,1,NULL);
INSERT INTO "Festivals_Performances" VALUES(3,1,NULL);
INSERT INTO "Festivals_Performances" VALUES(4,1,NULL);
INSERT INTO "Festivals_Performances" VALUES(5,1,NULL);
INSERT INTO "Festivals_Performances" VALUES(5,4,NULL);
CREATE TABLE Performances(
    PerformanceID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT,
    Description TEXT);
INSERT INTO "Performances" VALUES(1,'Live_Band','A live music performance featuring multiple bands.');
INSERT INTO "Performances" VALUES(2,'DJ','An electronic dance music DJ set.');
INSERT INTO "Performances" VALUES(3,'Comedy Show','A stand-up comedy showcase.');
INSERT INTO "Performances" VALUES(4,'Theatre','A professional theatre production.');
INSERT INTO "Performances" VALUES(5,'Circus','A traditional circus performance.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Festivals',5);
INSERT INTO "sqlite_sequence" VALUES('Performances',5);
INSERT INTO "sqlite_sequence" VALUES('Artists',5);
COMMIT;
