BEGIN TRANSACTION;
CREATE TABLE Artists(
    ArtistID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT ,
    Genre TEXT
);
INSERT INTO "Artists" VALUES(1,'John Doe',32,'Pop');
INSERT INTO "Artists" VALUES(2,'Jane Smith',28,'Rock');
INSERT INTO "Artists" VALUES(3,'Michael Johnson',45,'Hip Hop');
INSERT INTO "Artists" VALUES(4,'Emma Jones',30,'Country');
INSERT INTO "Artists" VALUES(5,'Olivia Brown',26,'Electronic');
CREATE TABLE Events(
    EventID INTEGER PRIMARY KEY,
    Name TEXT,
    Date DATE,
    Location TEXT
);
INSERT INTO "Events" VALUES(1,'Event1','2022-09-01','New York');
INSERT INTO "Events" VALUES(2,'Event2','2022-09-05','Los Angeles');
INSERT INTO "Events" VALUES(3,'Event3','2022-09-10','Chicago');
INSERT INTO "Events" VALUES(4,'Event4','2022-09-15','Boston');
INSERT INTO "Events" VALUES(5,'Event5','2022-09-20','Washington DC');
CREATE TABLE Participants(
    ParticipantID INTEGER PRIMARY KEY,
    ArtistID INTEGER REFERENCES Artists(ArtistID) ON DELETE SET NULL,
    PerformanceID INTEGER REFERENCES Performances(PerformanceID) ON DELETE CASCADE,
    Role TEXT
);
INSERT INTO "Participants" VALUES(1,1,1,'Lead Singer');
INSERT INTO "Participants" VALUES(2,2,1,'Backup Vocals');
INSERT INTO "Participants" VALUES(3,3,2,'Dancer');
INSERT INTO "Participants" VALUES(4,4,3,'Band Leader');
INSERT INTO "Participants" VALUES(5,5,5,'Host');
CREATE TABLE Performances(
    PerformanceID INTEGER PRIMARY KEY,
    EventID INTEGER REFERENCES Events(EventID) ON DELETE CASCADE,
    Type TEXT,
    AudienceSize INT ,
    PerformanceEvent VARCHAR(50),
    AuditionRoom TEXT
);
INSERT INTO "Performances" VALUES(1,1,'Concert',500,'Dance Recital','Room 101');
INSERT INTO "Performances" VALUES(2,2,'Competition',300,'Talent Show','Room 101');
INSERT INTO "Performances" VALUES(3,3,'Festival',400,'Dance Recital','Room 102');
INSERT INTO "Performances" VALUES(4,4,'Gala',200,'Talent Show','Room 103');
INSERT INTO "Performances" VALUES(5,5,'Showcase',150,'Open Mic Night','Room 101');
COMMIT;
