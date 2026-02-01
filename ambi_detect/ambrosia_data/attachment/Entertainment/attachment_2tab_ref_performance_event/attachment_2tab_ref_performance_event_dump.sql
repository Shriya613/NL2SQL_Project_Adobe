BEGIN TRANSACTION;
CREATE TABLE Audition_Room(
    RoomID INTEGER PRIMARY KEY,
    RoomName TEXT UNIQUE
);
INSERT INTO "Audition_Room" VALUES(1,'Room 101');
INSERT INTO "Audition_Room" VALUES(2,'Room 202');
CREATE TABLE "Dance_Recital"(
    RecitalID INTEGER PRIMARY KEY, `name` TEXT,
    Date DATE,
    Time TIME,
    RoomID INTEGER,
    FOREIGN KEY (RoomID) REFERENCES Audition_Room(RoomID)
);
INSERT INTO "Dance_Recital" VALUES(1,'Celestial Movements: A Balletic Odyssey','2023-07-01','19:00',1);
INSERT INTO "Dance_Recital" VALUES(2,'Rhythmic Revolutions','2023-08-15','20:00',2);
CREATE TABLE Participant(
    ParticipantID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Age INT ,
    PhoneNumber TEXT,
    Email TEXT,
    Role VARCHAR(255),
    RecitalID INTEGER,
    ShowID INTEGER,
    FOREIGN KEY (RecitalID) REFERENCES Dance_Recital(RecitalID),
    FOREIGN KEY (ShowID) REFERENCES Talent_Show(ShowID)
);
INSERT INTO "Participant" VALUES(1,'Jane','Smith',25,'555-1234','janesmith@email.com','Dancer',1,1);
INSERT INTO "Participant" VALUES(2,'John','Doe',30,'555-5678','john.doe@email.com','Singer',2,2);
INSERT INTO "Participant" VALUES(3,'Emily','Johnson',22,'555-8910','emily.johnson@email.com','Musician',1,1);
INSERT INTO "Participant" VALUES(4,'Michael','Brown',28,'555-1314','michaelbrown@email.com','Actor',2,2);
INSERT INTO "Participant" VALUES(5,'Olivia','Wilson',18,'555-2323','oliviawilson@email.com','Comedian',1,1);
CREATE TABLE Talent_Show(
    ShowID INTEGER PRIMARY KEY,
    Name VARCHAR(255),
    Description TEXT,
    RoomID INTEGER,
    FOREIGN KEY (RoomID) REFERENCES Audition_Room(RoomID)
);
INSERT INTO "Talent_Show" VALUES(1,'Summer Talent Extravaganza','A showcase of local talent.',1);
INSERT INTO "Talent_Show" VALUES(2,'Fall Variety Night','An evening of diverse acts and performances.',2);
COMMIT;
