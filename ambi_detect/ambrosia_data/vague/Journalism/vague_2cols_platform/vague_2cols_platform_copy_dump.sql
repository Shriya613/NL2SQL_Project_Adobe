BEGIN TRANSACTION;
CREATE TABLE ArticleCategoryRelationship(
    articleId INTEGER,
    categoryId INTEGER,
    FOREIGN KEY(articleId) REFERENCES NewsArticle(id),
    FOREIGN KEY(categoryId) REFERENCES Category(id));
INSERT INTO "ArticleCategoryRelationship" VALUES(1,1);
INSERT INTO "ArticleCategoryRelationship" VALUES(2,2);
CREATE TABLE Author(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Email TEXT UNIQUE);
INSERT INTO "Author" VALUES(1,'John Doe','johndoe@email.com');
INSERT INTO "Author" VALUES(2,'Jane Smith','janesmith@email.com');
CREATE TABLE Category(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT);
INSERT INTO "Category" VALUES(1,'Environment');
INSERT INTO "Category" VALUES(2,'Technology');
CREATE TABLE Comments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    authorEmail TEXT,
    articleId INTEGER,
    FOREIGN KEY(articleId) REFERENCES NewsArticle(id));
INSERT INTO "Comments" VALUES(1,'Great read!','commenter1@email.com',1);
INSERT INTO "Comments" VALUES(2,'Interesting insights.','commenter2@email.com',2);
CREATE TABLE "NewsArticle" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    Platform TEXT,
    News_Agency TEXT);
INSERT INTO "NewsArticle" VALUES(1,'Climate Change and Its Impact on Agriculture','This article discusses...','www.example.com','Global Times');
INSERT INTO "NewsArticle" VALUES(2,'The Rise of Electric Vehicles','In this article, we explore...','www.examplenews.com','EcoTimes');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Author',2);
INSERT INTO "sqlite_sequence" VALUES('Category',2);
INSERT INTO "sqlite_sequence" VALUES('Comments',2);
INSERT INTO "sqlite_sequence" VALUES('NewsArticle',2);
COMMIT;
