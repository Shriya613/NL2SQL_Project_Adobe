BEGIN TRANSACTION;
CREATE TABLE Articles(
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    PublishDate DATE,
    AuthorID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Authors(AuthorID));
INSERT INTO "Articles" VALUES(1,'Article 1','Content of article 1...','2022-01-01',1);
INSERT INTO "Articles" VALUES(2,'Article 2','Content of article 2...','2022-01-02',2);
INSERT INTO "Articles" VALUES(3,'Article 3','Content of article 3...','2022-01-03',3);
INSERT INTO "Articles" VALUES(4,'Article 4','Content of article 4...','2022-01-04',4);
INSERT INTO "Articles" VALUES(5,'Article 5','Content of article 5...','2022-01-05',5);
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    BirthDate DATE);
INSERT INTO "Authors" VALUES(1,'John','Smith','2000-01-01');
INSERT INTO "Authors" VALUES(2,'Jane','Doe','1998-06-15');
INSERT INTO "Authors" VALUES(3,'Michael','Brown','1987-12-25');
INSERT INTO "Authors" VALUES(4,'Emily','White','1995-09-23');
INSERT INTO "Authors" VALUES(5,'David','Black','2003-03-04');
CREATE TABLE DistributionPlatforms(
    PlatformID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Type TEXT );
INSERT INTO "DistributionPlatforms" VALUES(1,'Newspaper','Print Media');
INSERT INTO "DistributionPlatforms" VALUES(2,'Magazine','Print Media');
INSERT INTO "DistributionPlatforms" VALUES(3,'Website','Online Media');
INSERT INTO "DistributionPlatforms" VALUES(4,'YouTube Channel','Online Media');
CREATE TABLE Publications(
    PublicationID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT);
INSERT INTO "Publications" VALUES(1,'Pub A','Description of pub A.');
INSERT INTO "Publications" VALUES(2,'Pub B','Description of pub B.');
INSERT INTO "Publications" VALUES(3,'Pub C','Description of pub C.');
INSERT INTO "Publications" VALUES(4,'Pub D','Description of pub D.');
INSERT INTO "Publications" VALUES(5,'Pub E','Description of pub E.');
CREATE TABLE "PublishesIn"( `id` INTEGER,
    PublicationID INTEGER,
    PlatformID INTEGER,
    AudienceSize INTEGER DEFAULT 1,
    PRIMARY KEY(PublicationID, PlatformID),
    FOREIGN KEY(PublicationID) REFERENCES Publications(PublicationID),
    FOREIGN KEY(PlatformID) REFERENCES DistributionPlatforms(PlatformID));
INSERT INTO "PublishesIn" VALUES(1,1,1,300000);
INSERT INTO "PublishesIn" VALUES(2,2,1,200000);
INSERT INTO "PublishesIn" VALUES(3,3,2,100000);
INSERT INTO "PublishesIn" VALUES(4,4,2,50000);
INSERT INTO "PublishesIn" VALUES(5,5,3,100000);
INSERT INTO "PublishesIn" VALUES(6,6,4,200000);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('DistributionPlatforms',4);
INSERT INTO "sqlite_sequence" VALUES('Publications',5);
COMMIT;
