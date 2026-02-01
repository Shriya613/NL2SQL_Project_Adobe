BEGIN TRANSACTION;
CREATE TABLE Jurisdictions (
    jurisdiction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    jurisdiction_name TEXT
);
INSERT INTO "Jurisdictions" VALUES(1,'Federal Courts');
INSERT INTO "Jurisdictions" VALUES(2,'State Courts');
INSERT INTO "Jurisdictions" VALUES(3,'Appellate Courts');
INSERT INTO "Jurisdictions" VALUES(4,'District Courts');
INSERT INTO "Jurisdictions" VALUES(5,'Supreme Court');
CREATE TABLE Legal_Affairs_Reporters (
    reporter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Legal_Affairs_Reporters" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Legal_Affairs_Reporters" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Legal_Affairs_Reporters" VALUES(3,'Michael Brown','michael@example.com');
INSERT INTO "Legal_Affairs_Reporters" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Legal_Affairs_Reporters" VALUES(5,'David Johnson','david@example.com');
CREATE TABLE Legal_Affairs_Reporters_Jurisdictions (
    legal_affairs_reporter_id INTEGER,
    jurisdiction_id INTEGER,
    FOREIGN KEY(legal_affairs_reporter_id) REFERENCES Legal_Affairs_Reporters(reporter_id),
    FOREIGN KEY(jurisdiction_id) REFERENCES Jurisdictions(jurisdiction_id)
);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(1,1);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(2,3);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(3,4);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(4,2);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(5,6);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(2,1);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(3,1);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(4,1);
INSERT INTO "Legal_Affairs_Reporters_Jurisdictions" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Legal_Affairs_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Jurisdictions',5);
COMMIT;
