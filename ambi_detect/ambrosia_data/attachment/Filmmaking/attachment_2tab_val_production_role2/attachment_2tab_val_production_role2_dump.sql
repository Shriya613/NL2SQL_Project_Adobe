BEGIN TRANSACTION;
CREATE TABLE Actors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    RolesPlayed TEXT,
    KnownFor TEXT);
CREATE TABLE Casting_Directors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Salary REAL,
    Experience INT,
    Awards TEXT);
INSERT INTO "Casting_Directors" VALUES(1,'John Doe',80000.0,12,'Oscar Winner');
INSERT INTO "Casting_Directors" VALUES(2,'Jane Smith',90000.0,7,NULL);
INSERT INTO "Casting_Directors" VALUES(3,'Emily Johnson',70000.0,5,'Golden Globe Recipient');
CREATE TABLE Cinematographers(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    EquipmentUsed TEXT,
    Style TEXT,
    FilmsShot INT);
INSERT INTO "Cinematographers" VALUES(1,'Chris Black','Arri Alexa Mini, Red Epic Dragon','Naturalistic',50);
INSERT INTO "Cinematographers" VALUES(2,'Samantha White','Alexa 65, IMAX Cameras','Epic Landscapes',35);
INSERT INTO "Cinematographers" VALUES(3,'Peter Gray','Canon C300 Mark II, Sony A7S II','Documentary',40);
CREATE TABLE Location_Scouts(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Salary REAL,
    YearsExperience INT,
    NotableLocations TEXT);
INSERT INTO "Location_Scouts" VALUES(1,'Tom Brown',80000.0,15,'New York City, Paris, London');
INSERT INTO "Location_Scouts" VALUES(2,'Anna Williams',60000.0,10,'Rome, Sydney, Tokyo');
INSERT INTO "Location_Scouts" VALUES(3,'Michael Thompson',75000.0,12,'Los Angeles, Moscow, Beijing');
CREATE TABLE Producers(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Budget REAL,
    Projects INT,
    FamousFor TEXT);
INSERT INTO "Producers" VALUES(1,'David Miller',100000000.0,20,'Blockbuster Movies');
INSERT INTO "Producers" VALUES(2,'Lucy Davis',85000000.0,15,'Independent Film Producer');
INSERT INTO "Producers" VALUES(3,'Paul Green',120000000.0,25,'Action Movie Specialist');
CREATE TABLE Screenwriters(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Genre TEXT,
    ScriptsWritten INT,
    Nominations INT);
INSERT INTO "Screenwriters" VALUES(1,'Katherine Brown','Drama',20,5);
INSERT INTO "Screenwriters" VALUES(2,'Mark James','Comedy',30,7);
INSERT INTO "Screenwriters" VALUES(3,'Laura King','Thriller/Horror',15,3);
COMMIT;
