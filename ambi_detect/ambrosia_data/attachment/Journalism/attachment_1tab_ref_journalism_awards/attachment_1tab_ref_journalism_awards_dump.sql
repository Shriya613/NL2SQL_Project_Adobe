BEGIN TRANSACTION;
CREATE TABLE Articles(
    ArticleID INTEGER PRIMARY KEY,
    Title TEXT,
    Content TEXT,
    DatePublished DATETIME,
    AuthorID INTEGER,
    PublisherID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Authors(AuthorID),
    FOREIGN KEY(PublisherID) REFERENCES Publications(PublisherID));
INSERT INTO "Articles" VALUES(1,'The Rise of AI','AI has revolutionized many industries...','2023-06-15',1,1);
INSERT INTO "Articles" VALUES(2,'Climate Change Crisis','Human activities have caused climate change...','2023-06-16',2,2);
INSERT INTO "Articles" VALUES(3,'Technology Trends','The future of technology looks promising...','2023-06-17',3,3);
INSERT INTO "Articles" VALUES(4,'Economic Outlook','Global economy faces challenges ahead...','2023-06-18',4,4);
INSERT INTO "Articles" VALUES(5,'Political Updates','Latest political news from around the world...','2023-06-19',5,5);
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT);
INSERT INTO "Authors" VALUES(1,'John','Doe');
INSERT INTO "Authors" VALUES(2,'Jane','Smith');
INSERT INTO "Authors" VALUES(3,'Bob','Johnson');
INSERT INTO "Authors" VALUES(4,'Mary','Williams');
INSERT INTO "Authors" VALUES(5,'Tom','Brown');
CREATE TABLE Awards(
    AwardID INTEGER PRIMARY KEY,
    AwardType TEXT);
INSERT INTO "Awards" VALUES(1,'Pulitzer Prize');
INSERT INTO "Awards" VALUES(2,'George Polk Award');
CREATE TABLE Publications(
    PublisherID INTEGER PRIMARY KEY,
    Name TEXT,
    YearOfEstablishment INT);
INSERT INTO "Publications" VALUES(1,'New York Times',1851);
INSERT INTO "Publications" VALUES(2,'Washington Post',1877);
INSERT INTO "Publications" VALUES(3,'Los Angeles Times',1881);
INSERT INTO "Publications" VALUES(4,'Chicago Tribune',1847);
INSERT INTO "Publications" VALUES(5,'USA Today',1982);
CREATE TABLE "Winners"(
    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
    WinningArticleID INTEGER,
    Year INTEGER,
    AwardID INTEGER,
    FOREIGN KEY(WinningArticleID) REFERENCES Articles(ArticleID),
    FOREIGN KEY(AwardID) REFERENCES Awards(AwardID));
INSERT INTO "Winners" VALUES(1,1,2010,1);
INSERT INTO "Winners" VALUES(2,2,2011,1);
INSERT INTO "Winners" VALUES(3,3,2010,2);
INSERT INTO "Winners" VALUES(4,4,2012,2);
INSERT INTO "Winners" VALUES(5,5,2013,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Winners',5);
COMMIT;
