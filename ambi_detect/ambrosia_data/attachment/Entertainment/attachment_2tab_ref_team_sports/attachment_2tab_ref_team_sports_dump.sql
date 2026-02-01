BEGIN TRANSACTION;
CREATE TABLE Fitness_Level(
    ID INTEGER PRIMARY KEY,
    Level TEXT);
INSERT INTO "Fitness_Level" VALUES(1,'Elite');
INSERT INTO "Fitness_Level" VALUES(2,'Intermediate');
CREATE TABLE Football_Players(
    ID INTEGER PRIMARY KEY,
    PlayerName TEXT,
    Team TEXT,
    Position TEXT,
    Age INTEGER,
    Fitness_ID INTEGER,
    FOREIGN KEY (Fitness_ID) REFERENCES Fitness_Level(ID));
INSERT INTO "Football_Players" VALUES(1,'John Doe','Manchester United','Striker',30,1);
INSERT INTO "Football_Players" VALUES(2,'Jane Smith','Liverpool','Defender',28,2);
INSERT INTO "Football_Players" VALUES(3,'Tom Brown','Chelsea','Goalkeeper',32,1);
INSERT INTO "Football_Players" VALUES(4,'Emma Johnson','Arsenal','Midfielder',26,2);
INSERT INTO "Football_Players" VALUES(5,'William Davis','Tottenham Hotspur','Winger',29,1);
CREATE TABLE Rugby_Players(
    ID INTEGER PRIMARY KEY,
    PlayerName TEXT,
    Team TEXT,
    Position TEXT,
    Age INTEGER,
    Fitness_ID INTEGER,
    FOREIGN KEY (Fitness_ID) REFERENCES Fitness_Level(ID));
INSERT INTO "Rugby_Players" VALUES(1,'Michael Jones','London Irish','Prop',31,1);
INSERT INTO "Rugby_Players" VALUES(2,'Lucy Thompson','Harlequins','Scrumhalf',27,2);
INSERT INTO "Rugby_Players" VALUES(3,'David Williams','Saracens','Center',30,1);
INSERT INTO "Rugby_Players" VALUES(4,'Olivia Taylor','Gloucester','Flanker',26,2);
INSERT INTO "Rugby_Players" VALUES(5,'Ethan Young','Wasps','Lock',32,1);
COMMIT;
