BEGIN TRANSACTION;
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    written_by INTEGER REFERENCES Authors(id),
    Written_journalism TEXT,
    word_count TEXT,
    FOREIGN KEY (word_count) REFERENCES WordCounts(column));
INSERT INTO "Articles" VALUES(1,'First News Article',1,'News Article','1,000 words');
INSERT INTO "Articles" VALUES(2,'Second News Article',2,'News Article','800 words');
INSERT INTO "Articles" VALUES(3,'First Feature Article',3,'Feature Article','1,000 words');
INSERT INTO "Articles" VALUES(4,'Second Feature Article',4,'Feature Article','900 words');
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Authors" VALUES(1,'John Doe');
INSERT INTO "Authors" VALUES(2,'Jane Smith');
CREATE TABLE Comments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment TEXT,
    article_id INTEGER REFERENCES Articles(id));
INSERT INTO "Comments" VALUES(1,'Great article!',1);
INSERT INTO "Comments" VALUES(2,'Interesting insights.',2);
INSERT INTO "Comments" VALUES(3,'Well-written piece.',3);
INSERT INTO "Comments" VALUES(4,'Enjoyed reading this.',4);
CREATE TABLE Publications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER REFERENCES Articles(id),
    publisher_id INTEGER REFERENCES Publishers(id),
    publication_date DATE);
INSERT INTO "Publications" VALUES(1,1,1,'2023-01-01');
INSERT INTO "Publications" VALUES(2,2,2,'2023-01-02');
INSERT INTO "Publications" VALUES(3,3,1,'2023-01-03');
INSERT INTO "Publications" VALUES(4,4,2,'2023-01-04');
CREATE TABLE Publishers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Publishers" VALUES(1,'Newspaper XYZ');
INSERT INTO "Publishers" VALUES(2,'Magazine ABC');
CREATE TABLE WordCounts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column TEXT UNIQUE);
INSERT INTO "WordCounts" VALUES(1,'1,000 words');
INSERT INTO "WordCounts" VALUES(2,'750 words');
INSERT INTO "WordCounts" VALUES(3,'1,200 words');
INSERT INTO "WordCounts" VALUES(4,'800 words');
INSERT INTO "WordCounts" VALUES(5,'1,100 words');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',2);
INSERT INTO "sqlite_sequence" VALUES('Articles',4);
INSERT INTO "sqlite_sequence" VALUES('Publishers',2);
INSERT INTO "sqlite_sequence" VALUES('Publications',4);
INSERT INTO "sqlite_sequence" VALUES('Comments',4);
INSERT INTO "sqlite_sequence" VALUES('WordCounts',5);
COMMIT;
