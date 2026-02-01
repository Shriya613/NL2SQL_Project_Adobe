BEGIN TRANSACTION;
CREATE TABLE newsrooms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
CREATE TABLE publications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publishDate DATETIME,
    associatedStoryId INTEGER,
    FOREIGN KEY (associatedStoryId) REFERENCES storyIdeas(id));
INSERT INTO "publications" VALUES(1,'The Climate Crisis','Effects of climate change on human life...','2021-03-16 16:39:07',1);
INSERT INTO "publications" VALUES(2,'AI Revolution','Artificial Intelligence has changed our lives like never before...','2021-03-16 16:39:07',2);
INSERT INTO "publications" VALUES(3,'Vaccination Drive','How the global vaccination drive against COVID...','2021-03-16 16:39:07',3);
INSERT INTO "publications" VALUES(4,'New Horizons in Space','Astronomers have discovered new celestial bodies...','2021-03-16 16:39:07',4);
INSERT INTO "publications" VALUES(5,'Post Pandemic Economy','What does a post-COVID economy look like? ...','2021-03-16 16:39:07',5);
CREATE TABLE readershipStats(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    publicationId INTEGER,
    numPageViews INTEGER DEFAULT 0,
    avgTimeSpentReading NUMERIC DEFAULT 0,
    FOREIGN KEY (publicationId) REFERENCES publications(id));
INSERT INTO "readershipStats" VALUES(1,1,200,10);
INSERT INTO "readershipStats" VALUES(2,2,400,20);
INSERT INTO "readershipStats" VALUES(3,3,150,5);
INSERT INTO "readershipStats" VALUES(4,4,300,15);
INSERT INTO "readershipStats" VALUES(5,5,250,12);
CREATE TABLE reporters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    role TEXT, -- Can include Reporter, Photographer, Editor, Manager etc.,
    deadline DATETIME, -- Deadline for tasks/stories
    assigned_newsroom INTEGER,
    FOREIGN KEY (assigned_newsroom) REFERENCES newsrooms(id));
INSERT INTO "reporters" VALUES(1,'John Doe','Reporter','2021-03-18 16:30:00',1);
INSERT INTO "reporters" VALUES(2,'Jane Smith','Reporter','2021-03-22 16:30:00',2);
INSERT INTO "reporters" VALUES(3,'Mike Johnson','Photographer','2021-03-18 16:30:00',3);
INSERT INTO "reporters" VALUES(4,'Anna Brown','Photographer','2021-03-22 16:30:00',4);
CREATE TABLE storyIdeas(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    pitchDate DATETIME,
    assignedTo INTEGER,
    status TEXT ,
    FOREIGN KEY (assignedTo) REFERENCES reporters(id));
INSERT INTO "storyIdeas" VALUES(1,'Climate Change Impact','Impact of climate change on coastal cities','2021-03-15 16:39:07',1,NULL);
INSERT INTO "storyIdeas" VALUES(2,'Technology Advancement','Latest advancements in artificial intelligence','2021-03-15 16:39:07',2,NULL);
INSERT INTO "storyIdeas" VALUES(3,'Covid Vaccine Rollout','Global rollout of Covid vaccine','2021-03-15 16:39:07',3,NULL);
INSERT INTO "storyIdeas" VALUES(4,'Space Exploration','Recent achievements in space exploration','2021-03-15 16:39:07',4,NULL);
INSERT INTO "storyIdeas" VALUES(5,'Economic Recovery','Strategies for economic recovery post pandemic','2021-03-15 16:39:07',5,NULL);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('reporters',4);
INSERT INTO "sqlite_sequence" VALUES('storyIdeas',5);
INSERT INTO "sqlite_sequence" VALUES('publications',5);
INSERT INTO "sqlite_sequence" VALUES('readershipStats',5);
COMMIT;
