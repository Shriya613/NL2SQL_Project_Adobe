BEGIN TRANSACTION;
CREATE TABLE Category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
CREATE TABLE "Reporter" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT);
INSERT INTO "Reporter" VALUES(1,'John','Doe');
INSERT INTO "Reporter" VALUES(2,'Jane','Smith');
INSERT INTO "Reporter" VALUES(3,'Michael','Brown');
INSERT INTO "Reporter" VALUES(4,'Emily','White');
INSERT INTO "Reporter" VALUES(5,'William','Green');
CREATE TABLE "Story" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publishedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    journalistId INTEGER,
    categoryId INTEGER,
    FOREIGN KEY(journalistId) REFERENCES "Reporter"(id),
    FOREIGN KEY(categoryId) REFERENCES Category(id));
INSERT INTO "Story" VALUES(1,'Climate Change Summit','President discusses global warming at climate change summit.','2024-02-19 15:15:46',1,1);
INSERT INTO "Story" VALUES(2,'Economic Growth Slows Down','Unemployment rates rise as economic growth slows down.','2024-02-19 15:15:46',2,2);
INSERT INTO "Story" VALUES(3,'Tech Innovations','Latest tech innovations make life easier for millions of people.','2024-02-19 15:15:46',3,3);
INSERT INTO "Story" VALUES(4,'Health Crisis','Government struggles to contain health crisis.','2024-02-19 15:15:46',4,4);
INSERT INTO "Story" VALUES(5,'Space Exploration Advancements','NASA announces new breakthrough in space exploration technology.','2024-02-19 15:15:46',5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Reporter',5);
INSERT INTO "sqlite_sequence" VALUES('Story',5);
COMMIT;
