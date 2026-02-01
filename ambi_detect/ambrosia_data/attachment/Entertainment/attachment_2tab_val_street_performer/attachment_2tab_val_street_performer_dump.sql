BEGIN TRANSACTION;
CREATE TABLE Magicians(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Street_Location TEXT,
    MagicType TEXT,
    YearsExperience INT);
INSERT INTO "Magicians" VALUES(1,'Harry Houdini','Times Square, New York','Escape Artist',10);
INSERT INTO "Magicians" VALUES(2,'David Copperfield','Las Vegas','Illusionist',38);
CREATE TABLE Mimes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Street_Location TEXT,
    StyleOfMiming TEXT,
    PerformanceDuration INT);
INSERT INTO "Mimes" VALUES(1,'Marcel Marceau','Times Square, New York','Classic',20);
INSERT INTO "Mimes" VALUES(2,'Whitey McConnell','Boston','Character Miming',15);
CREATE TABLE Performers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Age INT,
    Genre TEXT);
INSERT INTO "Performers" VALUES(1,'Christopher','Columbus',35,'Juggling');
INSERT INTO "Performers" VALUES(2,'Carlos','Leon',28,'Fire Eating');
CREATE TABLE Shows(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ShowName TEXT,
    StartTime DATETIME,
    EndTime DATETIME,
    LocationID INT,
    FOREIGN KEY (LocationID) REFERENCES Venues(id));
INSERT INTO "Shows" VALUES(1,'The Illusionists','2022-07-15 19:00:00','2022-07-15 21:00:00',1);
INSERT INTO "Shows" VALUES(2,'Cirque du Soleil','2022-07-16 19:00:00','2022-07-16 21:00:00',2);
CREATE TABLE Venues(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    VenueName TEXT,
    Address TEXT,
    City TEXT,
    Capacity INT);
INSERT INTO "Venues" VALUES(1,'Madison Square Garden','New York, NY','New York',20000);
INSERT INTO "Venues" VALUES(2,'Radio City Music Hall','New York, NY','New York',6000);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Magicians',2);
INSERT INTO "sqlite_sequence" VALUES('Mimes',2);
INSERT INTO "sqlite_sequence" VALUES('Shows',2);
INSERT INTO "sqlite_sequence" VALUES('Venues',2);
INSERT INTO "sqlite_sequence" VALUES('Performers',2);
COMMIT;
