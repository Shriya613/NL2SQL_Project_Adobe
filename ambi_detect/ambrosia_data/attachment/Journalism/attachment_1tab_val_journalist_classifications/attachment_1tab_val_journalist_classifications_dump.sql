BEGIN TRANSACTION;
CREATE TABLE Articles (
    ArticleID INTEGER PRIMARY KEY,
    Title TEXT,
    Content TEXT,
    PublishDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    AuthorID INTEGER,
    FOREIGN KEY(AuthorID) REFERENCES Authors(AuthorID) ON DELETE SET NULL
);
INSERT INTO "Articles" VALUES(1,'Breaking News: Earth Shake','A powerful earthquake hits California','2024-03-07 23:25:22',1);
INSERT INTO "Articles" VALUES(2,'Tech Giants Merge','Apple and Google announce a merger','2024-03-07 23:25:22',2);
CREATE TABLE Authors (
    AuthorID INTEGER PRIMARY KEY,
    Name TEXT,
    Email TEXT,
    PhoneNumber VARCHAR(15),
    Classification TEXT ,
    Payment INTEGER
);
INSERT INTO "Authors" VALUES(1,'John Doe','johndoe@example.com','+1234567890','Stringer',2000);
INSERT INTO "Authors" VALUES(2,'Jane Smith','janesmith@example.com','+0987654321','Stringer',3000);
INSERT INTO "Authors" VALUES(3,'Alice Johnson','alicejohnson@example.com','+1357924680','Editor',2000);
INSERT INTO "Authors" VALUES(4,'Bob Brown','bobbrown@example.com','+2468135790','Editor',2500);
CREATE TABLE NewsOutlets (
    OutletID INTEGER PRIMARY KEY,
    Name TEXT,
    Location TEXT,
    Website URL
);
INSERT INTO "NewsOutlets" VALUES(1,'Global Times','New York, USA','https://www.globaltimes.com');
INSERT INTO "NewsOutlets" VALUES(2,'European Gazette','London, UK','https://www.europeangazette.co.uk');
CREATE TABLE Sources (
    SourceID INTEGER PRIMARY KEY,
    Name TEXT,
    ContactInfo TEXT,
    Reliability RATING , -- where 1 is least reliable and 5 is most reliable
    LastUsed DATETIME
);
INSERT INTO "Sources" VALUES(1,'Washington Post','washpost@example.com',4,'2021-07-01 12:00:00');
INSERT INTO "Sources" VALUES(2,'Reuters','reuters@example.com',5,'2021-07-15 15:30:00');
CREATE TABLE Submissions (
    SubmissionID INTEGER PRIMARY KEY,
    ArticleID INTEGER,
    OutletID INTEGER,
    Accepted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY(ArticleID) REFERENCES Articles(ArticleID) ON DELETE CASCADE,
    FOREIGN KEY(OutletID) REFERENCES NewsOutlets(OutletID) ON DELETE SET NULL
);
INSERT INTO "Submissions" VALUES(1,1,1,1);
INSERT INTO "Submissions" VALUES(2,2,2,0);
COMMIT;
