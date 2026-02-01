BEGIN TRANSACTION;
CREATE TABLE Articles(
    articleID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    authorName TEXT,
    publishDate DATETIME,
    content TEXT);
INSERT INTO "Articles" VALUES(1,'Britain Is Cutting Taxes Again. Why Now?','John Doe','2021-08-31 16:00:00','This is an important piece of news...');
INSERT INTO "Articles" VALUES(2,'Global Warming: The Human Impact','Jane Smith','2021-09-01 17:00:00','An insightful look at global warming caused by human activities...');
CREATE TABLE Interviews(
    interviewID INTEGER PRIMARY KEY AUTOINCREMENT,
    intervieweeName TEXT,
    interviewerName TEXT,
    interviewDate DATETIME,
    content TEXT,
    FK_articleID INTEGER,
    FOREIGN KEY (FK_articleID) REFERENCES Articles(articleID));
INSERT INTO "Interviews" VALUES(1,'Mark Johnson','Emily Brown','2021-08-31 20:00:00','Mr. Mark Johnson shared his insights about the tax cuts...',1);
INSERT INTO "Interviews" VALUES(2,'Linda Davis','Michael White','2021-09-01 21:00:00','Ms. Linda Davis discussed her views on climate change...',2);
CREATE TABLE PressReleases(
    pressReleaseID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseDate DATETIME,
    content TEXT,
    FK_articleID INTEGER,
    FOREIGN KEY (FK_articleID) REFERENCES Articles(articleID));
INSERT INTO "PressReleases" VALUES(1,'Q2 Earnings Report','2021-08-31 18:00:00','Our Q2 earnings report showcases ...',1);
INSERT INTO "PressReleases" VALUES(2,'Annual General Meeting Announcement','2021-09-01 19:00:00','We invite our shareholders to attend our annual general meeting on September 15th.',2);
CREATE TABLE Sources(
    sourceID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT,
    contactInfo TEXT);
INSERT INTO "Sources" VALUES(1,'New York Times','A leading daily newspaper founded in 1851...','+1 212-555-6789');
INSERT INTO "Sources" VALUES(2,'Washington Post','Founded in 1877, it is one of the largest newspapers in the US...','+1 202-334-6000');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Articles',2);
INSERT INTO "sqlite_sequence" VALUES('PressReleases',2);
INSERT INTO "sqlite_sequence" VALUES('Interviews',2);
INSERT INTO "sqlite_sequence" VALUES('Sources',2);
COMMIT;
