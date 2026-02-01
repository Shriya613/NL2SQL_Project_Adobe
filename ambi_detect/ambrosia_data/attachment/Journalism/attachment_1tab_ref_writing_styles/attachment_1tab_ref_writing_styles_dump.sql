BEGIN TRANSACTION;
CREATE TABLE ArticlePublications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ArticleId INTEGER,
    PublisherId INTEGER,
    PublishDate DATE,
    FOREIGN KEY(ArticleId) REFERENCES Articles(id),
    FOREIGN KEY(PublisherId) REFERENCES PublishingHouses(id)
);
INSERT INTO "ArticlePublications" VALUES(1,1,1,'2021-06-06');
INSERT INTO "ArticlePublications" VALUES(2,2,2,'2021-06-07');
INSERT INTO "ArticlePublications" VALUES(3,3,3,'2021-06-08');
INSERT INTO "ArticlePublications" VALUES(4,4,4,'2021-06-09');
INSERT INTO "ArticlePublications" VALUES(5,5,5,'2021-06-10');
CREATE TABLE Articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    PublicationDate DATE,
    AuthorId INTEGER,
    WritingStyleId INTEGER,
    FOREIGN KEY(AuthorId) REFERENCES Authors(id),
    FOREIGN KEY(WritingStyleId) REFERENCES WritingStyles(id)
);
INSERT INTO "Articles" VALUES(1,'The Short Story','Content of the short story...','2021-06-01',1,1);
INSERT INTO "Articles" VALUES(2,'Another Short Story','Content of another short story...','2021-06-02',2,1);
INSERT INTO "Articles" VALUES(3,'A Personal Essay','Content of a personal essay...','2021-06-03',1,2);
INSERT INTO "Articles" VALUES(4,'Yet Another Personal Essay','Content of yet another personal essay...','2021-06-04',3,2);
INSERT INTO "Articles" VALUES(5,'The Travelogue','Content of the travelogue...','2021-06-05',4,3);
CREATE TABLE Authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Authors" VALUES(1,'Evelyn O''Connor');
INSERT INTO "Authors" VALUES(2,'John Doe');
INSERT INTO "Authors" VALUES(3,'Jane Smith');
INSERT INTO "Authors" VALUES(4,'Robert Johnson');
INSERT INTO "Authors" VALUES(5,'Emily Davis');
CREATE TABLE PublishingHouses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "PublishingHouses" VALUES(1,'Random House');
INSERT INTO "PublishingHouses" VALUES(2,'Penguin Books');
INSERT INTO "PublishingHouses" VALUES(3,'HarperCollins');
INSERT INTO "PublishingHouses" VALUES(4,'Simon & Schuster');
INSERT INTO "PublishingHouses" VALUES(5,'Macmillan');
CREATE TABLE ReadingStats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ArticleId INTEGER,
    ReadCount INTEGER DEFAULT 0,
    FOREIGN KEY(ArticleId) REFERENCES Articles(id)
);
INSERT INTO "ReadingStats" VALUES(1,1,100);
INSERT INTO "ReadingStats" VALUES(2,2,150);
INSERT INTO "ReadingStats" VALUES(3,3,200);
INSERT INTO "ReadingStats" VALUES(4,4,250);
INSERT INTO "ReadingStats" VALUES(5,5,300);
CREATE TABLE WritingStyles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    StyleName TEXT UNIQUE
);
INSERT INTO "WritingStyles" VALUES(1,'Short Story');
INSERT INTO "WritingStyles" VALUES(2,'Personal Essay');
INSERT INTO "WritingStyles" VALUES(3,'Travelogue');
INSERT INTO "WritingStyles" VALUES(4,'Opinion Piece');
INSERT INTO "WritingStyles" VALUES(5,'Feature Story');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('WritingStyles',5);
INSERT INTO "sqlite_sequence" VALUES('PublishingHouses',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('ArticlePublications',5);
INSERT INTO "sqlite_sequence" VALUES('ReadingStats',5);
COMMIT;
