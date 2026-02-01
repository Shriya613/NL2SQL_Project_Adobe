BEGIN TRANSACTION;
CREATE TABLE FitnessLevels(
    id INTEGER PRIMARY KEY,
    LevelName TEXT
);
INSERT INTO "FitnessLevels" VALUES(1,'Elite');
INSERT INTO "FitnessLevels" VALUES(2,'Intermediate');
INSERT INTO "FitnessLevels" VALUES(3,'Beginner');
CREATE TABLE PlayerStats(
    id INTEGER PRIMARY KEY,
    PlayerId INTEGER,
    FitnessLevelId INTEGER,
    StatsText TEXT,
    FOREIGN KEY (PlayerId) REFERENCES Players(id),
    FOREIGN KEY (FitnessLevelId) REFERENCES FitnessLevels(id)
);
INSERT INTO "PlayerStats" VALUES(1,1,1,'Average stats');
INSERT INTO "PlayerStats" VALUES(2,2,2,'Above average stats');
INSERT INTO "PlayerStats" VALUES(3,3,3,'Below average stats');
INSERT INTO "PlayerStats" VALUES(4,4,1,'Excellent stats');
INSERT INTO "PlayerStats" VALUES(5,5,2,'Impressive stats');
CREATE TABLE Players(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    TeamTypeId INTEGER,
    SportId INTEGER,
    FOREIGN KEY (TeamTypeId) REFERENCES Team_Type(id),
    FOREIGN KEY (SportId) REFERENCES Sports(id)
);
INSERT INTO "Players" VALUES(1,'John Doe',28,1,1);
INSERT INTO "Players" VALUES(2,'Jane Smith',26,2,2);
INSERT INTO "Players" VALUES(3,'Alice Brown',30,1,1);
INSERT INTO "Players" VALUES(4,'Bob Johnson',29,2,2);
INSERT INTO "Players" VALUES(5,'Charlie White',27,1,1);
CREATE TABLE Sports(
    id INTEGER PRIMARY KEY,
    SportName TEXT
);
INSERT INTO "Sports" VALUES(1,'Football');
INSERT INTO "Sports" VALUES(2,'Rugby');
CREATE TABLE Team_Type(
    id INTEGER PRIMARY KEY,
    TypeName TEXT
);
INSERT INTO "Team_Type" VALUES(1,'National');
INSERT INTO "Team_Type" VALUES(2,'Club');
INSERT INTO "Team_Type" VALUES(3,'Youth');
COMMIT;
