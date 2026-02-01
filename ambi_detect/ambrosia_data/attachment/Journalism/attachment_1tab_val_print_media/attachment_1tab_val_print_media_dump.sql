BEGIN TRANSACTION;
CREATE TABLE Articles(
    ArticleID INTEGER PRIMARY KEY,
    Title TEXT,
    Content TEXT,
    PublishDate DATE,
    AuthorID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID)
);
INSERT INTO "Articles" VALUES(1,'Article 1','This is an article about something...','2022-01-01',1);
INSERT INTO "Articles" VALUES(2,'Article 2','Another article with content...','2022-01-07',2);
INSERT INTO "Articles" VALUES(3,'Article 3','Yet another piece of writing...','2022-01-14',3);
INSERT INTO "Articles" VALUES(4,'Article 4','Here comes a new story!','2022-01-21',4);
INSERT INTO "Articles" VALUES(5,'Article 5','Final article for this week...','2022-01-28',5);
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT
);
INSERT INTO "Authors" VALUES(1,'John','Doe');
INSERT INTO "Authors" VALUES(2,'Jane','Smith');
INSERT INTO "Authors" VALUES(3,'Michael','Brown');
INSERT INTO "Authors" VALUES(4,'Emma','Wilson');
INSERT INTO "Authors" VALUES(5,'Tom','Hanks');
CREATE TABLE Publications(
    ID INTEGER PRIMARY KEY,
    Title TEXT,
    Print_Media TEXT ,
    Publication_Frequency TEXT 
);
INSERT INTO "Publications" VALUES(1,'The Daily News','Newspaper','Daily');
INSERT INTO "Publications" VALUES(2,'Financial Times','Newspaper','Weekly');
INSERT INTO "Publications" VALUES(3,'Vanity Fair','Magazine','Monthly');
INSERT INTO "Publications" VALUES(4,'Time Magazine','Magazine','Weekly');
CREATE TABLE PublicationsAuthors(
    PublicationID INTEGER,
    AuthorID INTEGER,
    FOREIGN KEY (PublicationID) REFERENCES Publications(ID),
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID)
);
INSERT INTO "PublicationsAuthors" VALUES(1,1);
INSERT INTO "PublicationsAuthors" VALUES(2,2);
INSERT INTO "PublicationsAuthors" VALUES(3,3);
INSERT INTO "PublicationsAuthors" VALUES(4,4);
INSERT INTO "PublicationsAuthors" VALUES(5,5);
INSERT INTO "PublicationsAuthors" VALUES(6,1);
INSERT INTO "PublicationsAuthors" VALUES(7,2);
INSERT INTO "PublicationsAuthors" VALUES(8,3);
INSERT INTO "PublicationsAuthors" VALUES(9,4);
INSERT INTO "PublicationsAuthors" VALUES(10,5);
COMMIT;
