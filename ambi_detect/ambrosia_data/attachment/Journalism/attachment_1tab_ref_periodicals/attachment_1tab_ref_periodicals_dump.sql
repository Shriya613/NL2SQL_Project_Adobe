BEGIN TRANSACTION;
CREATE TABLE ArticleCategory(
    article_id INTEGER,
    cat_id INTEGER,
    FOREIGN KEY (article_id) REFERENCES Articles(id),
    FOREIGN KEY (cat_id) REFERENCES Category(id));
INSERT INTO "ArticleCategory" VALUES(1,1);
INSERT INTO "ArticleCategory" VALUES(2,2);
INSERT INTO "ArticleCategory" VALUES(3,3);
INSERT INTO "ArticleCategory" VALUES(4,4);
INSERT INTO "ArticleCategory" VALUES(5,5);
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    pub_date DATE,
    author_id INTEGER,
    publication_type TEXT ,
    FOREIGN KEY (author_id) REFERENCES Authors(id));
INSERT INTO "Articles" VALUES(1,'Article Title 1','Content of Article 1','2022-01-01',1,'Print');
INSERT INTO "Articles" VALUES(2,'Article Title 2','Content of Article 2','2022-01-02',2,'Online');
INSERT INTO "Articles" VALUES(3,'Article Title 3','Content of Article 3','2022-01-03',3,'Print');
INSERT INTO "Articles" VALUES(4,'Article Title 4','Content of Article 4','2022-01-04',4,'Online');
INSERT INTO "Articles" VALUES(5,'Article Title 5','Content of Article 5','2022-01-05',5,'Print');
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phone TEXT,
    address TEXT);
INSERT INTO "Authors" VALUES(1,'John Doe','john@example.com','123-456-7890','123 Main St');
INSERT INTO "Authors" VALUES(2,'Jane Smith','jane@example.com','987-654-3210','456 Elm St');
INSERT INTO "Authors" VALUES(3,'Alice Brown','alice@example.com','555-555-5555','789 Oak St');
INSERT INTO "Authors" VALUES(4,'Bob Johnson','bob@example.com','111-222-3333','100 Pine St');
INSERT INTO "Authors" VALUES(5,'Charlie Davis','charlie@example.com','444-444-4444','200 Maple Ave');
CREATE TABLE Category(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    descr TEXT UNIQUE);
INSERT INTO "Category" VALUES(1,'Politics');
INSERT INTO "Category" VALUES(2,'Economy');
INSERT INTO "Category" VALUES(3,'Technology');
INSERT INTO "Category" VALUES(4,'Entertainment');
INSERT INTO "Category" VALUES(5,'Sports');
CREATE TABLE Publications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    circulation_numbers INT,
    location TEXT);
INSERT INTO "Publications" VALUES(1,'The Daily News',20000,'USA');
INSERT INTO "Publications" VALUES(2,'The Weekly Times',15000,'Canada');
INSERT INTO "Publications" VALUES(3,'Time Magazine',20000,'USA');
INSERT INTO "Publications" VALUES(4,'Newsweek',12000,'UK');
INSERT INTO "Publications" VALUES(5,'Business Today',10000,'India');
CREATE TABLE PublicationsPeriodicals(
    publication_id INTEGER,
    periodical_type TEXT ,
    FOREIGN KEY (publication_id) REFERENCES Publications(id));
INSERT INTO "PublicationsPeriodicals" VALUES(1,'Newspaper');
INSERT INTO "PublicationsPeriodicals" VALUES(2,'Newspaper');
INSERT INTO "PublicationsPeriodicals" VALUES(3,'Magazine');
INSERT INTO "PublicationsPeriodicals" VALUES(4,'Magazine');
INSERT INTO "PublicationsPeriodicals" VALUES(5,'Journal');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Publications',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Category',5);
COMMIT;
