BEGIN TRANSACTION;
CREATE TABLE Advertisements (
    AdvID INTEGER PRIMARY KEY,
    Description TEXT,
    Budget DECIMAL(10, 2) ,
    StartDate TIMESTAMP,
    EndDate TIMESTAMP,
    OutletID INTEGER,
    FOREIGN KEY (OutletID) REFERENCES NewsOutlets(OutletID)
);
INSERT INTO "Advertisements" VALUES(1,'Advertisement 1',1000,'2021-01-01 00:00:00','2021-01-31 23:59:59',1);
INSERT INTO "Advertisements" VALUES(2,'Advertisement 2',1500,'2021-02-01 00:00:00','2021-02-28 23:59:59',2);
CREATE TABLE Articles (
    ArticleID INTEGER PRIMARY KEY,
    Title TEXT,
    PublicationDate TIMESTAMP,
    WordCount INTEGER ,
    AuthorID INTEGER,
    FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID)
);
INSERT INTO "Articles" VALUES(1,'Sample Article 1','2021-01-01 00:00:00',500,1);
INSERT INTO "Articles" VALUES(2,'Sample Article 2','2021-02-01 00:00:00',750,1);
CREATE TABLE "Audience" (
    DemoID INTEGER PRIMARY KEY,
    Demographics VARCHAR(255),
    "Article" INT,
    FOREIGN KEY ("Article") REFERENCES Articles(ArticleID)
);
INSERT INTO "Audience" VALUES(1,'25-34',1);
INSERT INTO "Audience" VALUES(2,'35-44',2);
CREATE TABLE Authors (
    AuthorID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE
);
INSERT INTO "Authors" VALUES(1,'John','Doe','1980-06-01');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','1975-03-15');
CREATE TABLE NewsOutlets (
    OutletID INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    FoundedYear INTEGER
);
INSERT INTO "NewsOutlets" VALUES(1,'Newspaper A','City A',1900);
INSERT INTO "NewsOutlets" VALUES(2,'Newspaper B','City B',1950);
CREATE TABLE Publishes (
    PublishID INTEGER PRIMARY KEY,
    ArticleID INTEGER,
    OutletID INTEGER,
    FOREIGN KEY (ArticleID) REFERENCES Articles(ArticleID),
    FOREIGN KEY (OutletID) REFERENCES NewsOutlets(OutletID)
);
INSERT INTO "Publishes" VALUES(1,1,1);
INSERT INTO "Publishes" VALUES(2,2,2);
COMMIT;
