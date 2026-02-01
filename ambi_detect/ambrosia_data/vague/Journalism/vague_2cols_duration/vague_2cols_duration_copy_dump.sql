BEGIN TRANSACTION;
CREATE TABLE Articles (
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    PublishDate DATETIME,
    ViewCount INTEGER DEFAULT 0
);
INSERT INTO "Articles" VALUES(1,'Article One','This is the content of article one.','2024-02-19 14:58:28',100);
INSERT INTO "Articles" VALUES(2,'Article Two','Content of article two goes here.','2024-02-19 14:58:28',80);
CREATE TABLE Authors (
    AuthorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    EmailAddress TEXT UNIQUE
);
INSERT INTO "Authors" VALUES(1,'John','Doe','johndoe@example.com');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','janesmith@example.com');
CREATE TABLE "Operations" (
    OperationID INTEGER PRIMARY KEY AUTOINCREMENT,
    Duration INTEGER,
    MediaOutlet TEXT,
    Revenue FLOAT,
    Expenses FLOAT,
    ProfitMargin REAL
);
INSERT INTO "Operations" VALUES(1,10,'Newspaper X',100000.0,75000.0,0.0);
INSERT INTO "Operations" VALUES(2,5,'Radio Y',50000.0,30000.0,0.0);
CREATE TABLE Publications (
    PublisherID INTEGER PRIMARY KEY,
    Name TEXT,
    WebsiteURL TEXT,
    FoundedYear INTEGER
);
INSERT INTO "Publications" VALUES(1,'Publication A','https://www.publicationa.com',2000);
INSERT INTO "Publications" VALUES(2,'Publication B','https://www.publicationb.com',2001);
CREATE TABLE Readers (
    ReaderID INTEGER PRIMARY KEY,
    FullName TEXT,
    EmailAddress TEXT,
    RegistrationDate DATETIME
);
INSERT INTO "Readers" VALUES(1,'Alice Johnson','alicejohnson@example.com','2024-02-19 14:58:28');
INSERT INTO "Readers" VALUES(2,'Bob Smith','bobsmit@example.net','2024-02-19 14:58:28');
CREATE TABLE Subscriptions (
    SubscriptionID INTEGER PRIMARY KEY AUTOINCREMENT,
    ReaderID INTEGER,
    PublisherID INTEGER,
    StartDate DATETIME,
    EndDate DATETIME,
    Price DECIMAL(10,2),
    FOREIGN KEY (ReaderID) REFERENCES Readers(ReaderID),
    FOREIGN KEY (PublisherID) REFERENCES Publications(PublisherID)
);
INSERT INTO "Subscriptions" VALUES(1,1,1,'2024-02-19 14:58:28',NULL,9.99);
INSERT INTO "Subscriptions" VALUES(2,2,2,'2024-02-19 14:58:28',NULL,12.99);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Articles',2);
INSERT INTO "sqlite_sequence" VALUES('Subscriptions',2);
INSERT INTO "sqlite_sequence" VALUES('Operations',2);
COMMIT;
