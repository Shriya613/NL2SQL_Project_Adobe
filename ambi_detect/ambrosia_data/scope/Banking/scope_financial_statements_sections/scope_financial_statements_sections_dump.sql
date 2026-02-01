BEGIN TRANSACTION;
CREATE TABLE Financial_Statements(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Date DATE
);
INSERT INTO "Financial_Statements" VALUES(1,'Company A','2021-07-30');
INSERT INTO "Financial_Statements" VALUES(2,'Company B','2021-08-31');
INSERT INTO "Financial_Statements" VALUES(3,'Company C','2021-09-30');
INSERT INTO "Financial_Statements" VALUES(4,'Company D','2021-10-31');
INSERT INTO "Financial_Statements" VALUES(5,'Company E','2021-11-30');
CREATE TABLE Financial_Statements_Sections(
    Financial_Statement_id INTEGER,
    Section_id INTEGER,
    FOREIGN KEY(Financial_Statement_id) REFERENCES Financial_Statements(id),
    FOREIGN KEY(Section_id) REFERENCES Sections(id),
    PRIMARY KEY(Financial_Statement_id, Section_id)
);
INSERT INTO "Financial_Statements_Sections" VALUES(1,1);
INSERT INTO "Financial_Statements_Sections" VALUES(2,1);
INSERT INTO "Financial_Statements_Sections" VALUES(3,1);
INSERT INTO "Financial_Statements_Sections" VALUES(4,1);
INSERT INTO "Financial_Statements_Sections" VALUES(5,1);
INSERT INTO "Financial_Statements_Sections" VALUES(5,5);
CREATE TABLE Sections(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    SectionName TEXT
);
INSERT INTO "Sections" VALUES(1,'Balance Sheet');
INSERT INTO "Sections" VALUES(2,'Profit & Loss Statement');
INSERT INTO "Sections" VALUES(3,'Cash Flow Statement');
INSERT INTO "Sections" VALUES(4,'Equity Statement');
INSERT INTO "Sections" VALUES(5,'Additional Information');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Sections',5);
INSERT INTO "sqlite_sequence" VALUES('Financial_Statements',5);
COMMIT;
