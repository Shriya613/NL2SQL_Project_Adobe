BEGIN TRANSACTION;
CREATE TABLE Article (
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    publishedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    newsAgencyId INTEGER,
    FOREIGN KEY(newsAgencyId) REFERENCES News_Agency(id)
);
INSERT INTO "Article" VALUES(1,'Test Article 1','This is a test article.','2024-02-19 14:31:13',1);
INSERT INTO "Article" VALUES(2,'Test Article 2','Another test article.','2024-02-19 14:31:13',2);
INSERT INTO "Article" VALUES(3,'Test Article 3','Yet another test article.','2024-02-19 14:31:13',3);
INSERT INTO "Article" VALUES(4,'Test Article 4','And here is one more test article.','2024-02-19 14:31:13',4);
INSERT INTO "Article" VALUES(5,'Test Article 5','Last but not least, a test article!','2024-02-19 14:31:13',5);
CREATE TABLE Author (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Author" VALUES(1,'John','Doe');
INSERT INTO "Author" VALUES(2,'Jane','Smith');
INSERT INTO "Author" VALUES(3,'Michael','Johnson');
INSERT INTO "Author" VALUES(4,'Emily','Williams');
INSERT INTO "Author" VALUES(5,'David','Brown');
CREATE TABLE AuthorArticle (
    authorId INTEGER,
    articleId INTEGER,
    FOREIGN KEY(authorId) REFERENCES Author(id),
    FOREIGN KEY(articleId) REFERENCES Article(id)
);
INSERT INTO "AuthorArticle" VALUES(1,1);
INSERT INTO "AuthorArticle" VALUES(2,2);
INSERT INTO "AuthorArticle" VALUES(3,3);
INSERT INTO "AuthorArticle" VALUES(4,4);
INSERT INTO "AuthorArticle" VALUES(5,5);
CREATE TABLE "News_Agency" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    websiteId INTEGER,
    FOREIGN KEY(websiteId) REFERENCES "Platform"(id));
INSERT INTO "News_Agency" VALUES(1,'CNN',1);
INSERT INTO "News_Agency" VALUES(2,'BBC',2);
INSERT INTO "News_Agency" VALUES(3,'The Guardian',3);
INSERT INTO "News_Agency" VALUES(4,'Al Jazeera',4);
INSERT INTO "News_Agency" VALUES(5,'New York Times',5);
CREATE TABLE "Platform" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    url TEXT UNIQUE
);
INSERT INTO "Platform" VALUES(1,'CNN','https://www.cnn.com');
INSERT INTO "Platform" VALUES(2,'BBC','https://www.bbc.co.uk');
INSERT INTO "Platform" VALUES(3,'The Guardian','https://www.theguardian.com');
INSERT INTO "Platform" VALUES(4,'Al Jazeera','https://www.aljazeera.com');
INSERT INTO "Platform" VALUES(5,'New York Times','https://www.nytimes.com');
COMMIT;
