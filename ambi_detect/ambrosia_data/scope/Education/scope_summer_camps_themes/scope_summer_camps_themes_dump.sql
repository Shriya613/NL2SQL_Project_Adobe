BEGIN TRANSACTION;
CREATE TABLE Summer_Camp_Themes(
    summerCampId INTEGER REFERENCES Summer_Camps(id),
    themeId INTEGER REFERENCES Themes(id));
INSERT INTO "Summer_Camp_Themes" VALUES(1,1);
INSERT INTO "Summer_Camp_Themes" VALUES(1,2);
INSERT INTO "Summer_Camp_Themes" VALUES(2,3);
INSERT INTO "Summer_Camp_Themes" VALUES(2,4);
INSERT INTO "Summer_Camp_Themes" VALUES(3,1);
INSERT INTO "Summer_Camp_Themes" VALUES(3,5);
INSERT INTO "Summer_Camp_Themes" VALUES(4,2);
INSERT INTO "Summer_Camp_Themes" VALUES(5,3);
INSERT INTO "Summer_Camp_Themes" VALUES(2,1);
INSERT INTO "Summer_Camp_Themes" VALUES(4,1);
INSERT INTO "Summer_Camp_Themes" VALUES(5,1);
CREATE TABLE Summer_Camps(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campName TEXT,
    location TEXT,
    startDate DATE,
    endDate DATE);
INSERT INTO "Summer_Camps" VALUES(1,'Camp A','New York','2023-06-18','2023-07-02');
INSERT INTO "Summer_Camps" VALUES(2,'Camp B','Los Angeles','2023-06-25','2023-07-09');
INSERT INTO "Summer_Camps" VALUES(3,'Camp C','Chicago','2023-07-03','2023-07-17');
INSERT INTO "Summer_Camps" VALUES(4,'Camp D','Miami','2023-06-20','2023-07-10');
INSERT INTO "Summer_Camps" VALUES(5,'Camp E','Houston','2023-06-25','2023-07-11');
CREATE TABLE Themes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    themeName TEXT);
INSERT INTO "Themes" VALUES(1,'Sports');
INSERT INTO "Themes" VALUES(2,'Art');
INSERT INTO "Themes" VALUES(3,'Science');
INSERT INTO "Themes" VALUES(4,'Technology');
INSERT INTO "Themes" VALUES(5,'Mathematics');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Themes',5);
INSERT INTO "sqlite_sequence" VALUES('Summer_Camps',5);
COMMIT;
