BEGIN TRANSACTION;
CREATE TABLE Career_Pathways(
    PathwayID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT
);
INSERT INTO "Career_Pathways" VALUES(1,'Early Childhood','This pathway focuses on teaching young children.');
INSERT INTO "Career_Pathways" VALUES(2,'Primary School','This pathway prepares teachers for grades K-6.');
INSERT INTO "Career_Pathways" VALUES(3,'Secondary School','This pathway trains educators for grades 7-12.');
INSERT INTO "Career_Pathways" VALUES(4,'Special Education','This pathway equips teachers with skills to work with students with special needs.');
INSERT INTO "Career_Pathways" VALUES(5,'Adult Education','This pathway targets educating adult learners.');
CREATE TABLE Career_Pathways_Opportunities(
    CareerPathwayID INTEGER,
    OpportunityID INTEGER,
    FOREIGN KEY (CareerPathwayID) REFERENCES Career_Pathways(PathwayID),
    FOREIGN KEY (OpportunityID) REFERENCES Opportunities(OpportunityID),
    PRIMARY KEY (CareerPathwayID, OpportunityID)
);
INSERT INTO "Career_Pathways_Opportunities" VALUES(1,1);
INSERT INTO "Career_Pathways_Opportunities" VALUES(1,3);
INSERT INTO "Career_Pathways_Opportunities" VALUES(2,1);
INSERT INTO "Career_Pathways_Opportunities" VALUES(2,4);
INSERT INTO "Career_Pathways_Opportunities" VALUES(3,2);
INSERT INTO "Career_Pathways_Opportunities" VALUES(3,5);
INSERT INTO "Career_Pathways_Opportunities" VALUES(4,2);
INSERT INTO "Career_Pathways_Opportunities" VALUES(4,5);
INSERT INTO "Career_Pathways_Opportunities" VALUES(5,3);
INSERT INTO "Career_Pathways_Opportunities" VALUES(5,4);
INSERT INTO "Career_Pathways_Opportunities" VALUES(3,1);
INSERT INTO "Career_Pathways_Opportunities" VALUES(4,1);
INSERT INTO "Career_Pathways_Opportunities" VALUES(5,1);
CREATE TABLE Opportunities(
    OpportunityID INTEGER PRIMARY KEY AUTOINCREMENT,
    Value TEXT
);
INSERT INTO "Opportunities" VALUES(1,'Teaching');
INSERT INTO "Opportunities" VALUES(2,'Curriculum Development');
INSERT INTO "Opportunities" VALUES(3,'Research');
INSERT INTO "Opportunities" VALUES(4,'Administration');
INSERT INTO "Opportunities" VALUES(5,'Consultancy');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Career_Pathways',5);
INSERT INTO "sqlite_sequence" VALUES('Opportunities',5);
COMMIT;
