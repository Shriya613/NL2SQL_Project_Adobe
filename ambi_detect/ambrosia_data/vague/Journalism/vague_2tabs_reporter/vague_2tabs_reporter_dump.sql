BEGIN TRANSACTION;
CREATE TABLE Category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE);
CREATE TABLE Journalist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fullname TEXT);
INSERT INTO "Journalist" VALUES(1,'John Doe');
INSERT INTO "Journalist" VALUES(2,'Jane Smith');
INSERT INTO "Journalist" VALUES(3,'Michael Brown');
INSERT INTO "Journalist" VALUES(4,'Emily White');
INSERT INTO "Journalist" VALUES(5,'William Green');
CREATE TABLE News_Agency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "News_Agency" VALUES(1,'The Daily Times','New York');
INSERT INTO "News_Agency" VALUES(2,'The Global Gazette','London');
INSERT INTO "News_Agency" VALUES(3,'The National Herald','Washington D.C.');
INSERT INTO "News_Agency" VALUES(4,'The European Chronicle','Paris');
INSERT INTO "News_Agency" VALUES(5,'Asia Today','Tokyo');
CREATE TABLE Story (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publishedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    journalistId INTEGER,
    agencyId INTEGER,
    categoryId INTEGER,
    FOREIGN KEY(journalistId) REFERENCES Journalist(id),
    FOREIGN KEY(agencyId) REFERENCES News_Agency(id),
    FOREIGN KEY(categoryId) REFERENCES Category(id));
INSERT INTO "Story" VALUES(1,'Climate Change Summit','President discusses global warming at climate change summit.','2024-02-19 15:15:46',1,1,1);
INSERT INTO "Story" VALUES(2,'Economic Growth Slows Down','Unemployment rates rise as economic growth slows down.','2024-02-19 15:15:46',2,3,2);
INSERT INTO "Story" VALUES(3,'Tech Innovations','Latest tech innovations make life easier for millions of people.','2024-02-19 15:15:46',3,2,3);
INSERT INTO "Story" VALUES(4,'Health Crisis','Government struggles to contain health crisis.','2024-02-19 15:15:46',4,4,4);
INSERT INTO "Story" VALUES(5,'Space Exploration Advancements','NASA announces new breakthrough in space exploration technology.','2024-02-19 15:15:46',5,5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Journalist',5);
INSERT INTO "sqlite_sequence" VALUES('News_Agency',5);
INSERT INTO "sqlite_sequence" VALUES('Story',5);
COMMIT;
