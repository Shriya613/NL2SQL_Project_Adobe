BEGIN TRANSACTION;
CREATE TABLE Home_Inspection_Issues(
    home_inspection_id INTEGER,
    issue_id INTEGER,
    FOREIGN KEY (home_inspection_id) REFERENCES Home_Inspections(id),
    FOREIGN KEY (issue_id) REFERENCES Issues(id));
INSERT INTO "Home_Inspection_Issues" VALUES(1,1);
INSERT INTO "Home_Inspection_Issues" VALUES(2,1);
INSERT INTO "Home_Inspection_Issues" VALUES(3,1);
INSERT INTO "Home_Inspection_Issues" VALUES(4,1);
INSERT INTO "Home_Inspection_Issues" VALUES(5,1);
INSERT INTO "Home_Inspection_Issues" VALUES(5,3);
CREATE TABLE Home_Inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    inspection_date DATE);
INSERT INTO "Home_Inspections" VALUES(1,'123 Main St','2022-01-01');
INSERT INTO "Home_Inspections" VALUES(2,'456 Elm St','2022-01-02');
INSERT INTO "Home_Inspections" VALUES(3,'789 Oak St','2022-01-03');
INSERT INTO "Home_Inspections" VALUES(4,'135 Maple Ave','2022-01-04');
INSERT INTO "Home_Inspections" VALUES(5,'987 Pine Rd','2022-01-05');
CREATE TABLE Issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_type TEXT);
INSERT INTO "Issues" VALUES(1,'Roof Damage');
INSERT INTO "Issues" VALUES(2,'Foundation Crack');
INSERT INTO "Issues" VALUES(3,'Electrical Problem');
INSERT INTO "Issues" VALUES(4,'Plumbing Issue');
INSERT INTO "Issues" VALUES(5,'Pest Infestation');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Issues',5);
INSERT INTO "sqlite_sequence" VALUES('Home_Inspections',5);
COMMIT;
