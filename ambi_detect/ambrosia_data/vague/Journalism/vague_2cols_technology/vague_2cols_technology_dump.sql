BEGIN TRANSACTION;
CREATE TABLE ArticleSources (
    articleSourceID INTEGER PRIMARY KEY AUTOINCREMENT,
    articleID INTEGER,
    sourceID INTEGER,
    referenceDate DATETIME,
    FOREIGN KEY(articleID) REFERENCES Articles(articleID),
    FOREIGN KEY(sourceID) REFERENCES Sources(sourceID)
);
INSERT INTO "ArticleSources" VALUES(1,1,1,'2021-07-01 10:00:00');
INSERT INTO "ArticleSources" VALUES(2,2,2,'2021-07-01 10:00:00');
INSERT INTO "ArticleSources" VALUES(3,3,3,'2021-07-01 10:00:00');
INSERT INTO "ArticleSources" VALUES(4,4,4,'2021-07-01 10:00:00');
INSERT INTO "ArticleSources" VALUES(5,5,5,'2021-07-01 10:00:00');
CREATE TABLE Articles (
    articleID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    authorID INTEGER,
    FOREIGN KEY(authorID) REFERENCES Authors(authorID)
);
INSERT INTO "Articles" VALUES(1,'First Article Title','This is the first article content.','2024-02-19 14:35:51',1);
INSERT INTO "Articles" VALUES(2,'Second Article Title','This is the second article content.','2024-02-19 14:35:51',2);
INSERT INTO "Articles" VALUES(3,'Third Article Title','This is the third article content.','2024-02-19 14:35:51',3);
INSERT INTO "Articles" VALUES(4,'Fourth Article Title','This is the fourth article content.','2024-02-19 14:35:51',4);
INSERT INTO "Articles" VALUES(5,'Fifth Article Title','This is the fifth article content.','2024-02-19 14:35:51',5);
CREATE TABLE Authors (
    authorID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Authors" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Authors" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Authors" VALUES(3,'Michael Johnson','michaeljohnson@example.com');
INSERT INTO "Authors" VALUES(4,'Emily Brown','emilybrown@example.com');
INSERT INTO "Authors" VALUES(5,'David Wilson','davidwilson@example.com');
CREATE TABLE ContentManagementSystems (
    cmsId INTEGER PRIMARY KEY AUTOINCREMENT,
    Content_Management_System TEXT,
    Web_Framework TEXT,
    Online_News_Platform TEXT,
    FOREIGN KEY(cmsId) REFERENCES Articles(authorID)
);
INSERT INTO "ContentManagementSystems" VALUES(1,'WordPress','PHP','Newspack');
INSERT INTO "ContentManagementSystems" VALUES(2,'Drupal','PHP','Tribune Content Agency');
INSERT INTO "ContentManagementSystems" VALUES(3,'Joomla','PHP','LimeSurvey');
INSERT INTO "ContentManagementSystems" VALUES(4,'Plone','Python','Pelagios');
INSERT INTO "ContentManagementSystems" VALUES(5,'Concrete5','PHP','Zen Cart');
CREATE TABLE EditorialTeamMembers (
    memberID INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    editorName TEXT
);
INSERT INTO "EditorialTeamMembers" VALUES(1,'Managing Editor','Mary Williams');
INSERT INTO "EditorialTeamMembers" VALUES(2,'Copy Chief','Tom Anderson');
INSERT INTO "EditorialTeamMembers" VALUES(3,'Features Editor','Ann Taylor');
INSERT INTO "EditorialTeamMembers" VALUES(4,'Sports Editor','Chris Harris');
INSERT INTO "EditorialTeamMembers" VALUES(5,'Opinion Editor','Laura King');
CREATE TABLE Sources (
    sourceID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    url TEXT UNIQUE
);
INSERT INTO "Sources" VALUES(1,'The New York Times','https://www.nytimes.com/');
INSERT INTO "Sources" VALUES(2,'CNN','https://www.cnn.com/');
INSERT INTO "Sources" VALUES(3,'BBC News','https://www.bbc.co.uk/news/');
INSERT INTO "Sources" VALUES(4,'Al Jazeera','https://www.aljazeera.com/');
INSERT INTO "Sources" VALUES(5,'Deutsche Welle','https://www.dw.com/');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Sources',5);
INSERT INTO "sqlite_sequence" VALUES('ArticleSources',5);
INSERT INTO "sqlite_sequence" VALUES('ContentManagementSystems',5);
INSERT INTO "sqlite_sequence" VALUES('EditorialTeamMembers',5);
COMMIT;
