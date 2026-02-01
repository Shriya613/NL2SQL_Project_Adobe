BEGIN TRANSACTION;
CREATE TABLE ArticleAuthors(
    articleId INTEGER REFERENCES Articles(id),
    journalistId INTEGER REFERENCES Journalists(id),
    PRIMARY KEY (articleId, journalistId)
);
INSERT INTO "ArticleAuthors" VALUES(1,1);
INSERT INTO "ArticleAuthors" VALUES(2,2);
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    publishDate DATE,
    editorId INTEGER REFERENCES Editors(id),
    FOREIGN KEY (editorId) REFERENCES Editors(id)
);
INSERT INTO "Articles" VALUES(1,'Article 1','Content of Article 1','2021-01-01',1);
INSERT INTO "Articles" VALUES(2,'Article 2','Content of Article 2','2021-01-15',2);
CREATE TABLE Editors(
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    phoneNumber TEXT,
    languages TEXT 
);
INSERT INTO "Editors" VALUES(1,'Michael Johnson','michael@example.com','+0987654321','English and Spanish');
INSERT INTO "Editors" VALUES(2,'Emily Davis','emily@example.com','+1234567890','Italian and Portuguese');
CREATE TABLE Journalists(
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    phoneNumber TEXT,
    languages TEXT 
);
INSERT INTO "Journalists" VALUES(1,'John Doe','john@example.com','+1234567890','English and Spanish');
INSERT INTO "Journalists" VALUES(2,'Jane Smith','jane@example.com','+9876543210','German and French');
CREATE TABLE Publications(
    id INTEGER PRIMARY KEY,
    title TEXT,
    publicationDate DATE
);
INSERT INTO "Publications" VALUES(1,'The Daily News','2021-01-01');
INSERT INTO "Publications" VALUES(2,'Weekly Times','2021-01-15');
CREATE TABLE Subscriptions(
    id INTEGER PRIMARY KEY,
    subscriberEmail TEXT,
    subscriptionType TEXT 
);
INSERT INTO "Subscriptions" VALUES(1,'subscription1@example.com','Premium');
INSERT INTO "Subscriptions" VALUES(2,'subscription2@example.com','Standard');
COMMIT;
