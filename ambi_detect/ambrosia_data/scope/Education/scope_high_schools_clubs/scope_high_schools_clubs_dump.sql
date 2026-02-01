BEGIN TRANSACTION;
CREATE TABLE Clubs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    club_name TEXT);
INSERT INTO "Clubs" VALUES(1,'Drama_Club');
INSERT INTO "Clubs" VALUES(2,'Math_Club');
INSERT INTO "Clubs" VALUES(3,'Art_Club');
INSERT INTO "Clubs" VALUES(4,'Music_Club');
INSERT INTO "Clubs" VALUES(5,'Chess_Club');
CREATE TABLE High_Schools(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "High_Schools" VALUES(1,'Central High School','New York');
INSERT INTO "High_Schools" VALUES(2,'East High School','Los Angeles');
INSERT INTO "High_Schools" VALUES(3,'South High School','Chicago');
INSERT INTO "High_Schools" VALUES(4,'West High School','San Francisco');
INSERT INTO "High_Schools" VALUES(5,'North High School','Seattle');
CREATE TABLE High_Schools_Clubs(
    high_school_id INTEGER,
    club_id INTEGER,
    FOREIGN KEY(high_school_id) REFERENCES High_Schools(id),
    FOREIGN KEY(club_id) REFERENCES Clubs(id));
INSERT INTO "High_Schools_Clubs" VALUES(1,1);
INSERT INTO "High_Schools_Clubs" VALUES(2,2);
INSERT INTO "High_Schools_Clubs" VALUES(3,3);
INSERT INTO "High_Schools_Clubs" VALUES(4,4);
INSERT INTO "High_Schools_Clubs" VALUES(5,5);
INSERT INTO "High_Schools_Clubs" VALUES(2,1);
INSERT INTO "High_Schools_Clubs" VALUES(3,1);
INSERT INTO "High_Schools_Clubs" VALUES(4,1);
INSERT INTO "High_Schools_Clubs" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('High_Schools',5);
INSERT INTO "sqlite_sequence" VALUES('Clubs',5);
COMMIT;
