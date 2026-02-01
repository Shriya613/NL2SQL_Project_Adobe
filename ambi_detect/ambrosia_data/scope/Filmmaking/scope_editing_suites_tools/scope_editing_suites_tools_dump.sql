BEGIN TRANSACTION;
CREATE TABLE Editing_Suites(
    SuiteID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Editing_Suites" VALUES(1,'Studio A','New York');
INSERT INTO "Editing_Suites" VALUES(2,'Studio B','Los Angeles');
INSERT INTO "Editing_Suites" VALUES(3,'Studio C','London');
INSERT INTO "Editing_Suites" VALUES(4,'Studio D','Toronto');
INSERT INTO "Editing_Suites" VALUES(5,'Studio E','Mumbai');
CREATE TABLE Editing_Suites_Tools(
    SuiteID INTEGER,
    ToolID INTEGER,
    FOREIGN KEY(SuiteID) REFERENCES Editing_Suites(SuiteID),
    FOREIGN KEY(ToolID) REFERENCES Tools(ToolID),
    PRIMARY KEY(SuiteID, ToolID));
INSERT INTO "Editing_Suites_Tools" VALUES(1,1);
INSERT INTO "Editing_Suites_Tools" VALUES(2,1);
INSERT INTO "Editing_Suites_Tools" VALUES(3,1);
INSERT INTO "Editing_Suites_Tools" VALUES(4,1);
INSERT INTO "Editing_Suites_Tools" VALUES(5,1);
INSERT INTO "Editing_Suites_Tools" VALUES(5,3);
CREATE TABLE Tools(
    ToolID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT);
INSERT INTO "Tools" VALUES(1,'Cutting_Software');
INSERT INTO "Tools" VALUES(2,'Color Grading Software');
INSERT INTO "Tools" VALUES(3,'Sound Design Software');
INSERT INTO "Tools" VALUES(4,'Visual Effects Software');
INSERT INTO "Tools" VALUES(5,'Animatronics');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Editing_Suites',5);
INSERT INTO "sqlite_sequence" VALUES('Tools',5);
COMMIT;
