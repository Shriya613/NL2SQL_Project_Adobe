BEGIN TRANSACTION;
CREATE TABLE Companies(
    company_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    location TEXT);
INSERT INTO "Companies" VALUES(1,'Company A','New York');
INSERT INTO "Companies" VALUES(2,'Company B','San Francisco');
INSERT INTO "Companies" VALUES(3,'Company C','Chicago');
INSERT INTO "Companies" VALUES(4,'Company D','Los Angeles');
INSERT INTO "Companies" VALUES(5,'Company E','Boston');
CREATE TABLE It_Support_Specialists(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INT);
INSERT INTO "It_Support_Specialists" VALUES(1,'John Doe',3);
INSERT INTO "It_Support_Specialists" VALUES(2,'Jane Smith',2);
INSERT INTO "It_Support_Specialists" VALUES(3,'Michael Johnson',10);
INSERT INTO "It_Support_Specialists" VALUES(4,'Emily Davis',6);
INSERT INTO "It_Support_Specialists" VALUES(5,'David Brown',8);
CREATE TABLE It_Support_Specialists_Systems(
    it_specialist_id INTEGER,
    system_id INTEGER,
    FOREIGN KEY(it_specialist_id) REFERENCES It_Support_Specialists(id),
    FOREIGN KEY(system_id) REFERENCES Systems(system_id));
INSERT INTO "It_Support_Specialists_Systems" VALUES(1,1);
INSERT INTO "It_Support_Specialists_Systems" VALUES(2,1);
INSERT INTO "It_Support_Specialists_Systems" VALUES(3,1);
INSERT INTO "It_Support_Specialists_Systems" VALUES(4,1);
INSERT INTO "It_Support_Specialists_Systems" VALUES(5,1);
INSERT INTO "It_Support_Specialists_Systems" VALUES(5,4);
CREATE TABLE Systems(
    system_id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_name TEXT,
    is_active BOOL DEFAULT TRUE);
INSERT INTO "Systems" VALUES(1,'Windows',1);
INSERT INTO "Systems" VALUES(2,'Linux',0);
INSERT INTO "Systems" VALUES(3,'MacOS',1);
INSERT INTO "Systems" VALUES(4,'Android',1);
INSERT INTO "Systems" VALUES(5,'iOS',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('It_Support_Specialists',5);
INSERT INTO "sqlite_sequence" VALUES('Systems',5);
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
COMMIT;
