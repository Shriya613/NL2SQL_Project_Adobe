BEGIN TRANSACTION;
CREATE TABLE Qualified_Tests(
    testerID INTEGER,
    testCaseID INTEGER,
    qualificationDate DATE,
    FOREIGN KEY (testerID) REFERENCES Quality_Assurance_Testers(testerID),
    FOREIGN KEY (testCaseID) REFERENCES Test_Cases(testCaseID));
INSERT INTO "Qualified_Tests" VALUES(1,1,NULL);
INSERT INTO "Qualified_Tests" VALUES(1,2,NULL);
INSERT INTO "Qualified_Tests" VALUES(2,1,NULL);
INSERT INTO "Qualified_Tests" VALUES(2,2,NULL);
INSERT INTO "Qualified_Tests" VALUES(3,1,NULL);
INSERT INTO "Qualified_Tests" VALUES(4,1,NULL);
INSERT INTO "Qualified_Tests" VALUES(5,1,NULL);
CREATE TABLE Quality_Assurance_Testers(
    testerID INTEGER PRIMARY KEY AUTOINCREMENT,
    fullName TEXT);
INSERT INTO "Quality_Assurance_Testers" VALUES(1,'John Doe');
INSERT INTO "Quality_Assurance_Testers" VALUES(2,'Jane Roe');
INSERT INTO "Quality_Assurance_Testers" VALUES(3,'Mike Smith');
INSERT INTO "Quality_Assurance_Testers" VALUES(4,'Anna Brown');
INSERT INTO "Quality_Assurance_Testers" VALUES(5,'David White');
CREATE TABLE Test_Cases(
    testCaseID INTEGER PRIMARY KEY AUTOINCREMENT,
    caseType TEXT,
    description TEXT);
INSERT INTO "Test_Cases" VALUES(1,'Smoke_Test','This is a smoke test case.');
INSERT INTO "Test_Cases" VALUES(2,'Functional_Test','This is a functional test case.');
INSERT INTO "Test_Cases" VALUES(3,'Integration_Test','This is an integration test case.');
INSERT INTO "Test_Cases" VALUES(4,'Regression_Test','This is a regression test case.');
INSERT INTO "Test_Cases" VALUES(5,'Performance_Test','This is a performance test case.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Quality_Assurance_Testers',5);
INSERT INTO "sqlite_sequence" VALUES('Test_Cases',5);
COMMIT;
