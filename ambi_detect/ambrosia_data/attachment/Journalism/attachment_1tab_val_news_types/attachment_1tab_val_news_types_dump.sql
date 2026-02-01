BEGIN TRANSACTION;
CREATE TABLE ArticleCategoryLinks(
    ArticleID INTEGER,
    CategoryID INTEGER,
    FOREIGN KEY(ArticleID) REFERENCES Articles(ArticleID),
    FOREIGN KEY(CategoryID) REFERENCES Categories(CategoryID),
    PRIMARY KEY(ArticleID, CategoryID)
);
INSERT INTO "ArticleCategoryLinks" VALUES(1,1);
INSERT INTO "ArticleCategoryLinks" VALUES(2,3);
INSERT INTO "ArticleCategoryLinks" VALUES(3,2);
INSERT INTO "ArticleCategoryLinks" VALUES(4,1);
INSERT INTO "ArticleCategoryLinks" VALUES(5,3);
INSERT INTO "ArticleCategoryLinks" VALUES(6,1);
INSERT INTO "ArticleCategoryLinks" VALUES(7,3);
INSERT INTO "ArticleCategoryLinks" VALUES(8,2);
INSERT INTO "ArticleCategoryLinks" VALUES(9,1);
INSERT INTO "ArticleCategoryLinks" VALUES(10,3);
CREATE TABLE Articles(
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    Event TEXT,
    PublicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    NewsType TEXT ,
    AuthorID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Authors(AuthorID)
);
INSERT INTO "Articles" VALUES(1,'Turkey–Syria Earthquakes Breaking News','This is a breaking news article about...','Turkey–Syria earthquakes','2024-03-07 23:34:04','Breaking News',1);
INSERT INTO "Articles" VALUES(2,'New Technologies Disrupt Market','This is an analysis of new technologies...','Technology Advancements','2024-03-07 23:34:04','Breaking News',2);
INSERT INTO "Articles" VALUES(3,'In Depth Look at Economic Impact of Earthquakes','A detailed examination of...','Turkey–Syria earthquakes','2024-03-07 23:34:04','News Analysis',1);
INSERT INTO "Articles" VALUES(4,'The Future of Sports Post Pandemic','An analysis of how sports have changed since...','Post-Pandemic Sports Development','2024-03-07 23:34:04','News Analysis',2);
INSERT INTO "Articles" VALUES(5,'Global Health Challenges','Exploring various global health challenges...','Global Health Initiatives','2024-03-07 23:34:04','General',3);
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FullName TEXT,
    EmailAddress TEXT UNIQUE
);
INSERT INTO "Authors" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Authors" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Authors" VALUES(3,'Alice Johnson','alicejohnson@example.com');
CREATE TABLE Keywords(
    KeywordID INTEGER PRIMARY KEY AUTOINCREMENT,
    KeywordText VARCHAR(255)
);
INSERT INTO "Keywords" VALUES(1,'Earthquake');
INSERT INTO "Keywords" VALUES(2,'Politics');
INSERT INTO "Keywords" VALUES(3,'Technology');
INSERT INTO "Keywords" VALUES(4,'Sports');
INSERT INTO "Keywords" VALUES(5,'Healthcare');
INSERT INTO "Keywords" VALUES(6,'Environment');
INSERT INTO "Keywords" VALUES(7,'Business');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',3);
INSERT INTO "sqlite_sequence" VALUES('Keywords',7);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
COMMIT;
