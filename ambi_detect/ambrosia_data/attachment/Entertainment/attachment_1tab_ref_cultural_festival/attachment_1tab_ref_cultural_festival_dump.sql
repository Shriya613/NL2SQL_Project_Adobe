BEGIN TRANSACTION;
CREATE TABLE "Events" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FestivalID INTEGER,
    "Attendance" INTEGER,
    ParticipantID INTEGER,
    FOREIGN KEY(FestivalID) REFERENCES Festivals(id),
    FOREIGN KEY(ParticipantID) REFERENCES Participants(id));
INSERT INTO "Events" VALUES(1,5,50000,1);
INSERT INTO "Events" VALUES(2,1,60000,2);
INSERT INTO "Events" VALUES(3,4,50000,3);
INSERT INTO "Events" VALUES(4,2,60000,4);
INSERT INTO "Events" VALUES(5,3,70000,5);
CREATE TABLE Festivals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Cultural_Festival TEXT,
    Location TEXT);
INSERT INTO "Festivals" VALUES(1,'Summer Music Festival','Music Festival','New York');
INSERT INTO "Festivals" VALUES(2,'Fall Film Festival','Film Festival','Los Angeles');
INSERT INTO "Festivals" VALUES(3,'Winter Jazz Concerts','Music Festival','Chicago');
INSERT INTO "Festivals" VALUES(4,'Spring Dance Showcase','Dance Festival','Boston');
INSERT INTO "Festivals" VALUES(5,'Autumn Art Exhibition','Art Festival','Miami');
CREATE TABLE Participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Role TEXT,
    Age INT);
INSERT INTO "Participants" VALUES(1,'John','Smith','Performer',32);
INSERT INTO "Participants" VALUES(2,'Jane','Doe','Volunteer',28);
INSERT INTO "Participants" VALUES(3,'Alice','Johnson','Organizer',37);
INSERT INTO "Participants" VALUES(4,'Bob','Williams','Technician',41);
INSERT INTO "Participants" VALUES(5,'Emma','Martin','Artist',29);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Festivals',5);
INSERT INTO "sqlite_sequence" VALUES('Participants',5);
INSERT INTO "sqlite_sequence" VALUES('Events',10);
COMMIT;
