BEGIN TRANSACTION;
CREATE TABLE Articles (
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Written_Journalism TEXT ,
    Word_Count TEXT, -- You may want to store this value as an integer instead if precision matters.
    PublicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    AuthorID INTEGER,
    CategoryID INTEGER,
    PublisherID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID),
    FOREIGN KEY (CategoryID) REFERENCES Categories(CategoryID),
    FOREIGN KEY (PublisherID) REFERENCES Publishers(PublisherID)
);
INSERT INTO "Articles" VALUES(1,'The US Election','News Article','2000 words','2024-03-07 23:54:05',1,1,1);
INSERT INTO "Articles" VALUES(2,'A New Economic Policy','News Article','3000 words','2024-03-07 23:54:05',2,2,2);
INSERT INTO "Articles" VALUES(3,'An In-Depth Look at Space Exploration','Feature Article','3000 words','2024-03-07 23:54:05',1,3,1);
INSERT INTO "Articles" VALUES(4,'Climate Change Effects on Wildlife','Feature Article','2000 words','2024-03-07 23:54:05',2,4,2);
CREATE TABLE Authors (
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT
);
INSERT INTO "Authors" VALUES(1,'John','Doe');
INSERT INTO "Authors" VALUES(2,'Jane','Smith');
CREATE TABLE Categories (
    CategoryID INTEGER PRIMARY KEY AUTOINCREMENT,
    CategoryName TEXT
);
INSERT INTO "Categories" VALUES(1,'Politics');
INSERT INTO "Categories" VALUES(2,'Economy');
CREATE TABLE Publishers (
    PublisherID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Publishers" VALUES(1,'New York Times');
INSERT INTO "Publishers" VALUES(2,'Washington Post');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',2);
INSERT INTO "sqlite_sequence" VALUES('Publishers',2);
INSERT INTO "sqlite_sequence" VALUES('Categories',2);
INSERT INTO "sqlite_sequence" VALUES('Articles',4);
COMMIT;
