BEGIN TRANSACTION;
CREATE TABLE ArticleCategories(
    article_id INTEGER,
    category_id INTEGER,
    FOREIGN KEY(article_id) REFERENCES Articles(id),
    FOREIGN KEY(category_id) REFERENCES Categories(id)
);
INSERT INTO "ArticleCategories" VALUES(1,1);
INSERT INTO "ArticleCategories" VALUES(2,2);
CREATE TABLE ArticleReporters(
    reporter_id INTEGER,
    article_id INTEGER,
    FOREIGN KEY(reporter_id) REFERENCES Reporters(id),
    FOREIGN KEY(article_id) REFERENCES Articles(id)
);
INSERT INTO "ArticleReporters" VALUES(1,1);
INSERT INTO "ArticleReporters" VALUES(2,2);
CREATE TABLE ArticleSources(
    source_id INTEGER,
    article_id INTEGER,
    FOREIGN KEY(source_id) REFERENCES Sources(id),
    FOREIGN KEY(article_id) REFERENCES Articles(id)
);
INSERT INTO "ArticleSources" VALUES(1,1);
INSERT INTO "ArticleSources" VALUES(2,2);
INSERT INTO "ArticleSources" VALUES(3,1);
INSERT INTO "ArticleSources" VALUES(4,2);
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    date DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "Articles" VALUES(1,'Article1','Content1','2024-03-07 23:09:43');
INSERT INTO "Articles" VALUES(2,'Article2','Content2','2024-03-07 23:09:43');
CREATE TABLE Categories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO "Categories" VALUES(1,'Category1');
INSERT INTO "Categories" VALUES(2,'Category2');
CREATE TABLE Reporters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Reporters" VALUES(1,'Reporter1','email1@example.com');
INSERT INTO "Reporters" VALUES(2,'Reporter2','email2@example.com');
CREATE TABLE Sources(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    event TEXT,
    person TEXT
);
INSERT INTO "Sources" VALUES(1,'Source1','Press Conference','Jack Layton');
INSERT INTO "Sources" VALUES(2,'Source2','Press Conference','John Doe');
INSERT INTO "Sources" VALUES(3,'Source3','Media Scrum','Jack Layton');
INSERT INTO "Sources" VALUES(4,'Source4','Media Scrum','Jane Smith');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Articles',2);
INSERT INTO "sqlite_sequence" VALUES('Reporters',2);
INSERT INTO "sqlite_sequence" VALUES('Sources',4);
INSERT INTO "sqlite_sequence" VALUES('Categories',2);
COMMIT;
