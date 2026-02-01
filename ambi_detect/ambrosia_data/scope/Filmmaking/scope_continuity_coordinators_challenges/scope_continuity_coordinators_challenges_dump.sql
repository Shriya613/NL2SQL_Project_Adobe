BEGIN TRANSACTION;
CREATE TABLE Challenges(
    ChallengeID INTEGER PRIMARY KEY,
    ChallengeName TEXT
);
INSERT INTO "Challenges" VALUES(1,'Wardrobe Changes');
INSERT INTO "Challenges" VALUES(2,'Lighting Changes');
INSERT INTO "Challenges" VALUES(3,'Prop Management');
INSERT INTO "Challenges" VALUES(4,'Camera Angles');
INSERT INTO "Challenges" VALUES(5,'Sound Quality');
CREATE TABLE Continuity_Coordinators(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Experience INT
);
INSERT INTO "Continuity_Coordinators" VALUES(1,'John Doe',3);
INSERT INTO "Continuity_Coordinators" VALUES(2,'Jane Smith',7);
INSERT INTO "Continuity_Coordinators" VALUES(3,'Alice Brown',8);
INSERT INTO "Continuity_Coordinators" VALUES(4,'Bob White',6);
INSERT INTO "Continuity_Coordinators" VALUES(5,'Carol Black',9);
CREATE TABLE Continuity_Coordinators_Challenges(
    CoordinatorID INTEGER,
    ChallengeID INTEGER,
    FOREIGN KEY (CoordinatorID) REFERENCES Continuity_Coordinators(ID),
    FOREIGN KEY (ChallengeID) REFERENCES Challenges(ChallengeID)
);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(1,1);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(1,2);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(2,3);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(3,4);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(4,5);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(5,1);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(5,2);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(2,1);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(3,1);
INSERT INTO "Continuity_Coordinators_Challenges" VALUES(4,1);
COMMIT;
