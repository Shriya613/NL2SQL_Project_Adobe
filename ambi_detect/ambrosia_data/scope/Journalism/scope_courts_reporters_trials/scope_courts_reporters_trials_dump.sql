BEGIN TRANSACTION;
CREATE TABLE Courts_Reporters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Courts_Reporters" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Courts_Reporters" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Courts_Reporters" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Courts_Reporters" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Courts_Reporters" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Courts_Reporters_Trials(
    reporterId INTEGER,
    trialId INTEGER,
    FOREIGN KEY (reporterId) REFERENCES Courts_Reporters(id),
    FOREIGN KEY (trialId) REFERENCES Trials(id));
INSERT INTO "Courts_Reporters_Trials" VALUES(1,1);
INSERT INTO "Courts_Reporters_Trials" VALUES(2,2);
INSERT INTO "Courts_Reporters_Trials" VALUES(3,3);
INSERT INTO "Courts_Reporters_Trials" VALUES(4,4);
INSERT INTO "Courts_Reporters_Trials" VALUES(5,5);
INSERT INTO "Courts_Reporters_Trials" VALUES(1,6);
INSERT INTO "Courts_Reporters_Trials" VALUES(2,7);
INSERT INTO "Courts_Reporters_Trials" VALUES(3,8);
INSERT INTO "Courts_Reporters_Trials" VALUES(4,9);
INSERT INTO "Courts_Reporters_Trials" VALUES(5,10);
INSERT INTO "Courts_Reporters_Trials" VALUES(2,1);
INSERT INTO "Courts_Reporters_Trials" VALUES(3,1);
INSERT INTO "Courts_Reporters_Trials" VALUES(4,1);
INSERT INTO "Courts_Reporters_Trials" VALUES(5,1);
CREATE TABLE Trials(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caseName TEXT,
    trialType TEXT);
INSERT INTO "Trials" VALUES(1,'High_profile_Criminal_Case_1','Criminal');
INSERT INTO "Trials" VALUES(2,'Other_Trial_1','Civil');
INSERT INTO "Trials" VALUES(3,'High_profile_Criminal_Case_2','Criminal');
INSERT INTO "Trials" VALUES(4,'Other_Trial_2','Civil');
INSERT INTO "Trials" VALUES(5,'High_profile_Criminal_Case_3','Criminal');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Courts_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Trials',5);
COMMIT;
