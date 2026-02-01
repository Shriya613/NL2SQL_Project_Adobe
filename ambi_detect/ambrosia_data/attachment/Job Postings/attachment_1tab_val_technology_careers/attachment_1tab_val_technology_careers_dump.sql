BEGIN TRANSACTION;
CREATE TABLE Companies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Companies" VALUES(1,'Company A','New York');
INSERT INTO "Companies" VALUES(2,'Company B','San Francisco');
INSERT INTO "Companies" VALUES(3,'Company C','Boston');
INSERT INTO "Companies" VALUES(4,'Company D','Chicago');
INSERT INTO "Companies" VALUES(5,'Company E','Los Angeles');
CREATE TABLE Roles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    Technology_Career TEXT ,
    Grade TEXT  );
INSERT INTO "Roles" VALUES(1,'Software Engineer I','Software Developer','Entry');
INSERT INTO "Roles" VALUES(2,'Software Engineer II','Software Developer','Middle');
INSERT INTO "Roles" VALUES(3,'Software Engineer III','Software Developer','Senior');
INSERT INTO "Roles" VALUES(4,'Data Scientist I','Data Analyst','Entry');
INSERT INTO "Roles" VALUES(5,'Data Scientist II','Data Analyst','Middle');
INSERT INTO "Roles" VALUES(6,'Data Science Manager','Data Analyst','Senior');
CREATE TABLE Technologies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    technology_name TEXT UNIQUE);
INSERT INTO "Technologies" VALUES(1,'JavaScript');
INSERT INTO "Technologies" VALUES(2,'Python');
INSERT INTO "Technologies" VALUES(3,'SQL');
INSERT INTO "Technologies" VALUES(4,'Ruby on Rails');
INSERT INTO "Technologies" VALUES(5,'GoLang');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Technologies',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',6);
COMMIT;
