BEGIN TRANSACTION;
CREATE TABLE Dance_Recitals(
    RecitalID INTEGER PRIMARY KEY,
    Name TEXT,
    Date DATE,
    Time TIME,
    Audition_Room TEXT
);
INSERT INTO "Dance_Recitals" VALUES(1,'Spring Fling','2023-03-27','19:00:00','Room 101');
INSERT INTO "Dance_Recitals" VALUES(2,'Autumn Spectacular','2023-10-15','20:00:00','Studio A');
CREATE TABLE Participation(
    ParticipantID INTEGER PRIMARY KEY,
    PerformerID INTEGER,
    Dance_RecitalID INTEGER,
    Talent_ShowID INTEGER,
    FOREIGN KEY(PerformerID) REFERENCES Performers(PerformerID),
    FOREIGN KEY(Dance_RecitalID) REFERENCES Dance_Recitals(RecitalID),
    FOREIGN KEY(Talent_ShowID) REFERENCES Talent_Shows(ShowID)
);
INSERT INTO "Participation" VALUES(1,1,1,1);
INSERT INTO "Participation" VALUES(2,2,1,2);
INSERT INTO "Participation" VALUES(3,3,2,1);
INSERT INTO "Participation" VALUES(4,4,NULL,2);
INSERT INTO "Participation" VALUES(5,5,NULL,1);
CREATE TABLE Performers(
    PerformerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Age INT,
    Role TEXT
);
INSERT INTO "Performers" VALUES(1,'John','Smith',18,'Dancer');
INSERT INTO "Performers" VALUES(2,'Jane','Doe',22,'Vocalist');
INSERT INTO "Performers" VALUES(3,'Michael','Brown',26,'Instrumentalist');
CREATE TABLE Talent_Shows(
    ShowID INTEGER PRIMARY KEY,
    Name TEXT,
    Theme TEXT,
    Location TEXT,
    Audition_Room TEXT
);
INSERT INTO "Talent_Shows" VALUES(1,'Rising Stars','Music Talents','Gymnasium','Room 101');
INSERT INTO "Talent_Shows" VALUES(2,'Creative Expressions','Artistic Talents','Conference Hall','Stage B');
COMMIT;
