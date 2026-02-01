BEGIN TRANSACTION;
CREATE TABLE "Assignments" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Origin INTEGER,
    FOREIGN KEY(Origin) REFERENCES Events(id),
    FOREIGN KEY(Origin) REFERENCES Photojournalists(id));
INSERT INTO "Assignments" VALUES(1,1);
INSERT INTO "Assignments" VALUES(2,2);
INSERT INTO "Assignments" VALUES(3,3);
INSERT INTO "Assignments" VALUES(4,4);
INSERT INTO "Assignments" VALUES(5,5);
CREATE TABLE Events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Location TEXT,
    Date DATE);
INSERT INTO "Events" VALUES(1,'Protest Rally','Downtown','2022-01-01');
INSERT INTO "Events" VALUES(2,'Election Day','City Hall','2022-11-08');
INSERT INTO "Events" VALUES(3,'Festival','Park','2022-07-15');
INSERT INTO "Events" VALUES(4,'Concert','Arena','2022-10-20');
INSERT INTO "Events" VALUES(5,'Marathon','Streets','2022-04-10');
CREATE TABLE Photojournalists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Birthdate DATE);
INSERT INTO "Photojournalists" VALUES(1,'John Doe','1980-06-23');
INSERT INTO "Photojournalists" VALUES(2,'Jane Smith','1991-07-05');
INSERT INTO "Photojournalists" VALUES(3,'Michael Johnson','1985-10-02');
INSERT INTO "Photojournalists" VALUES(4,'Emily Davis','1993-01-10');
INSERT INTO "Photojournalists" VALUES(5,'David Brown','1989-05-20');
CREATE TABLE Photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FilePath TEXT,
    Caption TEXT,
    TakenOn DATE);
INSERT INTO "Photos" VALUES(1,'/path/to/photo1.jpg','Riot police at protest rally','2022-01-01');
INSERT INTO "Photos" VALUES(2,'/path/to/photo2.jpg','Voters waiting to vote','2022-11-08');
INSERT INTO "Photos" VALUES(3,'/path/to/photo3.jpg','Crowd enjoying music','2022-07-15');
INSERT INTO "Photos" VALUES(4,'/path/to/photo4.jpg','Musician performing on stage','2022-10-20');
INSERT INTO "Photos" VALUES(5,'/path/to/photo5.jpg','Runners crossing finish line','2022-04-10');
CREATE TABLE PublishedPhotos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    AssignmentId INTEGER,
    PhotoId INTEGER,
    PublicationDate DATE,
    FOREIGN KEY(AssignmentId) REFERENCES Assignments(id),
    FOREIGN KEY(PhotoId) REFERENCES Photos(id));
INSERT INTO "PublishedPhotos" VALUES(1,1,1,'2022-01-02');
INSERT INTO "PublishedPhotos" VALUES(2,2,2,'2022-11-10');
INSERT INTO "PublishedPhotos" VALUES(3,3,3,'2022-07-20');
INSERT INTO "PublishedPhotos" VALUES(4,4,4,'2022-10-25');
INSERT INTO "PublishedPhotos" VALUES(5,5,5,'2022-04-15');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Photojournalists',5);
INSERT INTO "sqlite_sequence" VALUES('Events',5);
INSERT INTO "sqlite_sequence" VALUES('Photos',5);
INSERT INTO "sqlite_sequence" VALUES('PublishedPhotos',5);
INSERT INTO "sqlite_sequence" VALUES('Assignments',5);
COMMIT;
